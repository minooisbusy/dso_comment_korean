/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


/**
 * @brief 세 가지 누적 모드로 포인트를 추가한다.
 * @details mode 0(Active)      : `rJ->resF(현재 반복에서 계산된 최신 잔차)를 사용한다.
 * 			mode 1(Linearized)  : `r->res_toZeroF`(선형화 지점의 잔차)와 현재 상태와의 
 * 								  차이(`delta`)를 이용해 잔차를 근사.
 * 			mode 2(Marginalized): `r->res_toZeoF`(선형화 지점의 잔차)를 직접 사용한다.
 * @param p 추가될 포인트
 * @param ef 백엔드 시스템
 * @param tid target ID
 */
template<int mode>
void AccumulatedTopHessianSSE::addPoint(EFPoint* p, EnergyFunctional const * const ef, int tid)	// 0 = active, 1 = linearized, 2=marginalize
{


	assert(mode==0 || mode==1 || mode==2);

	VecCf dc = ef->cDeltaF; // 내부 파라미터의 차이
	float dd = p->deltaF;   // 역깊이의 차이

	float bd_acc=0;
	float Hdd_acc=0;
	VecCf  Hcd_acc = VecCf::Zero();

	for(EFResidual* r : p->residualsAll)
	{
		// 활성 포인트, 이미 선형화는된 포인트는 추가될 수 없다. 돌아가라.
		if(mode==0)
		{
			if(r->isLinearized || !r->isActive()) continue;
		}
		if(mode==1)
		{
			if(!r->isLinearized || !r->isActive()) continue;
		}
		if(mode==2)
		{
			if(!r->isActive()) continue;
			assert(r->isLinearized); // 잔차가 "절대" 선형화 되지 않는 상태라는것을 보증
		}


		// 3. Jacobian을 가져온다. Jacobian은 PointFrameResidual::linearize()에서 계산 된다.
		RawResidualJacobian* rJ = r->J; // 마지막 선형화 시점의 Jacobian J(x_0)
		int htIDX = r->hostIDX + r->targetIDX*nframes[tid]; //? host to target index
		Mat18f dp = ef->adHTdeltaF[htIDX]; // Δξ, Δa, Δb: 포즈 및 광도 파라미터의 "상대적" 변화량(Target frame에서 변화량)



		VecNRf resApprox; // 계산될 근사 잔차 r(x)를 저장할 변수; 최대 포인트 개수만큼의 크기
		if(mode==0)
			resApprox = rJ->resF;
		if(mode==2)
			resApprox = r->res_toZeroF;
		if(mode==1)
		{
			// 4. JΔx 항 계산 (SIMD를 위한 준비 단계)
			// d[u,v]/d [xi, c, idepth] = d[u,v]/d[xi] * delta[xi] + d[u,v]/d[c] * delta[c] + d[u,v]/d[idepth] * delta[idepth]
			// compute Jp*delta; 변수가 연산한 값을 인자로 받기에, 각 변수 값을 임시 객체에서 SIMD registor로 복사한다.
			// _mm_set1_ps는 네 개의 32bit float 값 모두 동일한 값을 복사한다. 즉, _mm_set1_ps(5.0f) -> [5.0f, 5.0f, 5.0f, 5.0f]
			__m128 Jp_delta_x = _mm_set1_ps(rJ->Jpdxi[0].dot(dp.head<6>())+rJ->Jpdc[0].dot(dc)+rJ->Jpdd[0]*dd);
			__m128 Jp_delta_y = _mm_set1_ps(rJ->Jpdxi[1].dot(dp.head<6>())+rJ->Jpdc[1].dot(dc)+rJ->Jpdd[1]*dd);
			
			// Δ[a]
			__m128 delta_a = _mm_set1_ps((float)(dp[6])); // dr/da * delta[a]
			//  Δ[b]
			__m128 delta_b = _mm_set1_ps((float)(dp[7])); // dr/db * delta[b]

			// 5. r(x) ≈ r(x₀) + JΔx 계산 (SIMD 연산)
			for(int i=0;i<patternNum;i+=4)
			{
				// PATTERN: rtz += resF - [JI*Jp Ja]*delta.
				__m128 rtz = _mm_load_ps(((float*)&r->res_toZeroF)+i); // r(x_0)를 할당

				// rJ->JIdx와 Jp_delta_x를 곱하면, 
				//d[r]/d[u,v] * d[u,v]/d[xi, c, idepth] * Δ[xi, c, idepth] 
				// = d[r]/d[xi, c, idepth] * Δ[xi, c, idepth]
				rtz = _mm_add_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx))+i),Jp_delta_x));
				rtz = _mm_add_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx+1))+i),Jp_delta_y));
				rtz = _mm_add_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF))+i),delta_a));
				rtz = _mm_add_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF+1))+i),delta_b));
				_mm_store_ps(((float*)&resApprox)+i, rtz); // cpu 변수인 resApprox로 저장
			}
		}

		// 6.b= -J*r(x_0)에서, J를 구성하기 위함이다.
		// need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
		Vec2f JI_r(0,0); // r(x_0) * d[I]/d[u,v]
		Vec2f Jab_r(0,0); // r(x_0) * d[i]/d[a,b]
		float rr=0;
		for(int i=0;i<patternNum;i++)
		{
			JI_r[0] += resApprox[i] *rJ->JIdx[0][i]; // r(x_0) * d[I]/d[u]
			JI_r[1] += resApprox[i] *rJ->JIdx[1][i]; // r(x_0) * d[I]/d[v]
			Jab_r[0] += resApprox[i] *rJ->JabF[0][i]; // r(x_0) * d[I]/d[a]
			Jab_r[1] += resApprox[i] *rJ->JabF[1][i]; // r(x_0) * d[i]/d[b]
			rr += resApprox[i]*resApprox[i]; //r^2
		}


		// H_cc
		acc[tid][htIDX].update( // 계산 H_cc ≈ (∂p/∂x_cam)ᵀ * ((∂r/∂p)ᵀ(∂r/∂p)) * (∂p/∂x_cam)
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				rJ->JIdx2(0,0),rJ->JIdx2(0,1),rJ->JIdx2(1,1));

		// H_ab, b_ab, r^2를 계산
		acc[tid][htIDX].updateBotRight(
				rJ->Jab2(0,0), rJ->Jab2(0,1), Jab_r[0],
				rJ->Jab2(1,1), Jab_r[1],rr); // rr은 b-vector 계산용이다.

		// H_cd
		acc[tid][htIDX].updateTopRight(
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				rJ->JabJIdx(0,0), rJ->JabJIdx(0,1),
				rJ->JabJIdx(1,0), rJ->JabJIdx(1,1),
				JI_r[0], JI_r[1]);


		Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;

		// bd_acc은 b=J^T*r이 아니라, J^T*r의 "일부"; b_d = d[r]/d[idepth]^T * r
		// bd_acc += r(x_0) * d[I]/d[u,v] * d[u, v] / d[idepth]
		bd_acc +=  JI_r[0]*rJ->Jpdd[0] + JI_r[1]*rJ->Jpdd[1]; // 즉, idepth에 대한 b만을 계산. for loop에 +=는 패턴 덧셈
		Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd); // idepth 부분에 
		Hcd_acc += rJ->Jpdc[0]*Ji2_Jpdd[0] + rJ->Jpdc[1]*Ji2_Jpdd[1];

		nres[tid]++;
	}

	if(mode==0)
	{
		p->Hdd_accAF = Hdd_acc;
		p->bd_accAF = bd_acc;
		p->Hcd_accAF = Hcd_acc;
	}
	if(mode==1 || mode==2)
	{
		p->Hdd_accLF = Hdd_acc;
		p->bd_accLF = bd_acc;
		p->Hcd_accLF = Hcd_acc;
	}
	if(mode==2)
	{
		p->Hcd_accAF.setZero();
		p->Hdd_accAF = 0;
		p->bd_accAF = 0;
	}

}
template void AccumulatedTopHessianSSE::addPoint<0>(EFPoint* p, EnergyFunctional const * const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint* p, EnergyFunctional const * const ef, int tid);
template void AccumulatedTopHessianSSE::addPoint<2>(EFPoint* p, EnergyFunctional const * const ef, int tid);








void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const * const EF, bool usePrior, bool useDelta, int tid)
{
	// 1. Hessian과 b-vector를 프레임 개수x(포즈+ab)(8) + 내부 파라미터 개수(4)의 크기로 할당하고 0으로 초기화 한다. 
	H = MatXX::Zero(nframes[tid]*8+CPARS, nframes[tid]*8+CPARS);
	b = VecX::Zero(nframes[tid]*8+CPARS);


	// 모든 프레임 쌍(host, target)에 대해 반복한다
	// 각 잔차(residual)는 host와 target 프레임을 연결하므로, 이 두 프레임의 상태에 모두 영향을 준다.
	for(int h=0;h<nframes[tid];h++)
		for(int t=0;t<nframes[tid];t++)
		{
			int hIdx = CPARS+h*8; // host 프레임 변수가 H, b에서 시작하는 인덱스
			int tIdx = CPARS+t*8; // target 프레임 변수가 H, b에서 시작하는 인덱스
			int aidx = h+nframes[tid]*t; // (h, t) 쌍에 대한 고유 인덱스

			// 3. (h, t) 쌍에 대해 누적된 정보(accH)를 가져온다.
			// acc[tid][aidx]는 addPoint 함수에서 계산 된 JᵀJ와 Jᵀr 정보를 담고 있다.
			acc[tid][aidx].finish(); // 전체 누적을 합한다.
			if(acc[tid][aidx].num==0) continue; // 누적된 정보 없으면 패스!

			MatPCPC accH = acc[tid][aidx].H.cast<double>(); // 13x13 크기의 누적된 행렬


			// 4. Adjoint 행렬을 사용하여 누적된 정보를 전체 H, b에 조립한다.
			// accH는 target 프레임의 지역 좌표계 기준 정보이므로,
			// Adjoint 행렬(EF->adHost, EF->adTarget)을 사용해 전역 좌표계로 변환하여 더해준다.
			// 이 과정은 J_global = J_local * Adjoint와 유사한 변환을 H행렬에 적용하는 것이다.

			// H 행렬의 대각 블록 (H_hh, H_tt) 업데이트
			H.block<8,8>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adHost[aidx].transpose();
			H.block<8,8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

			// H 행렬의 비대각 블록(H_ht) 업데이트
			H.block<8,8>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

			// H 행렬의 카메라 내부 파라미터-포즈 결합 블록 (H_ch, H_ct) 업데이트
			H.block<8,CPARS>(hIdx,0).noalias() += EF->adHost[aidx] * accH.block<8,CPARS>(CPARS,0);
			H.block<8,CPARS>(tIdx,0).noalias() += EF->adTarget[aidx] * accH.block<8,CPARS>(CPARS,0);

			//H 행렬의 카메라 내부 파라미터 블록 (H_cc) 업데이트
			H.topLeftCorner<CPARS,CPARS>().noalias() += accH.block<CPARS,CPARS>(0,0);

			// b-vector 업데이트
			b.segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8,1>(CPARS,8+CPARS);
			b.segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8,1>(CPARS,8+CPARS);
			b.head<CPARS>().noalias() += accH.block<CPARS,1>(0,8+CPARS);
		}


	// 5. 대칭성을 이용해 하삼각행렬을 채운다. 이전 단계는 상삼각행렬만 계산함!
	// ----- new: copy transposed parts.
	for(int h=0;h<nframes[tid];h++)
	{
		int hIdx = CPARS+h*8;
		H.block<CPARS,8>(0,hIdx).noalias() = H.block<8,CPARS>(hIdx,0).transpose();

		for(int t=h+1;t<nframes[tid];t++)
		{
			int tIdx = CPARS+t*8;
			H.block<8,8>(hIdx, tIdx).noalias() += H.block<8,8>(tIdx, hIdx).transpose();
			H.block<8,8>(tIdx, hIdx).noalias() = H.block<8,8>(hIdx, tIdx).transpose();
		}
	}


	if(usePrior)
	{
		assert(useDelta);
		H.diagonal().head<CPARS>() += EF->cPrior;
		b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
		for(int h=0;h<nframes[tid];h++)
		{
            H.diagonal().segment<8>(CPARS+h*8) += EF->frames[h]->prior;
            b.segment<8>(CPARS+h*8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
		}
	}
}


void AccumulatedTopHessianSSE::stitchDoubleInternal(
		MatXX* H, VecX* b, EnergyFunctional const * const EF, bool usePrior,
		int min, int max, Vec10* stats, int tid)
{
	int toAggregate = NUM_THREADS;
	if(tid == -1) { toAggregate = 1; tid = 0; }	// special case: if we dont do multithreading, dont aggregate.
	if(min==max) return;


	for(int k=min;k<max;k++)
	{
		int h = k%nframes[0];
		int t = k/nframes[0];

		int hIdx = CPARS+h*8;
		int tIdx = CPARS+t*8;
		int aidx = h+nframes[0]*t;

		assert(aidx == k);

		MatPCPC accH = MatPCPC::Zero();

		for(int tid2=0;tid2 < toAggregate;tid2++)
		{
			acc[tid2][aidx].finish();
			if(acc[tid2][aidx].num==0) continue;
			accH += acc[tid2][aidx].H.cast<double>();
		}

		H[tid].block<8,8>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adHost[aidx].transpose();

		H[tid].block<8,8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

		H[tid].block<8,8>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

		H[tid].block<8,CPARS>(hIdx,0).noalias() += EF->adHost[aidx] * accH.block<8,CPARS>(CPARS,0);

		H[tid].block<8,CPARS>(tIdx,0).noalias() += EF->adTarget[aidx] * accH.block<8,CPARS>(CPARS,0);

		H[tid].topLeftCorner<CPARS,CPARS>().noalias() += accH.block<CPARS,CPARS>(0,0);

		b[tid].segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8,1>(CPARS,CPARS+8);

		b[tid].segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8,1>(CPARS,CPARS+8);

		b[tid].head<CPARS>().noalias() += accH.block<CPARS,1>(0,CPARS+8);

	}


	// only do this on one thread.
	if(min==0 && usePrior)
	{
		H[tid].diagonal().head<CPARS>() += EF->cPrior;
		b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
		for(int h=0;h<nframes[tid];h++)
		{
            H[tid].diagonal().segment<8>(CPARS+h*8) += EF->frames[h]->prior;
            b[tid].segment<8>(CPARS+h*8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);

		}
	}
}



}
