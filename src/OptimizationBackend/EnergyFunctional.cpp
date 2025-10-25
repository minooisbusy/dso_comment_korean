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


#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;


void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
{

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;
	adHost = new Mat88[nFrames*nFrames]; // 8x8 행렬을 프레임 개수의 제곱 만큼 할당
	adTarget = new Mat88[nFrames*nFrames]; // 8x8 행렬을 프레임 개수의 제곱 만큼 할당

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			FrameHessian* host = frames[h]->data;
			FrameHessian* target = frames[t]->data;
			int idx = h + t * nFrames;

			// 오른쪽부터 읽어서, H->W->T로 가니, Host->Target 변환.
			SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

			Mat88 AH = Mat88f::Identity(); // 8 = 6 + 2, 6 for pose, 2 for photo. cal.
			Mat88 AT = Mat88f::Identity();

			// Host have to switch coordinate system to target. So Adjoint is needed.
			// $T_h$에 지역 섭동을 수행하여 $T_{ht}$에 섭동을 추가하면, 아래 꼴이 된다.
			// 의미: T_{ht}'가 T_{ht}에서 얼마만큼 변했는가?
			AH.topLeftCorner<6,6>() = -hostToTarget.Adj().transpose();
			// Target already in target frame. So, Identity mapping is valid.
			AT.topLeftCorner<6,6>() = Mat66::Identity();


			Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
			AT(6,6) = -affLL[0];
			AH(6,6) = affLL[0];
			AT(7,7) = -1;
			AH(7,7) = affLL[0];

			AH.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AH.block<3,8>(3,0) *= SCALE_XI_ROT;
			AH.block<1,8>(6,0) *= SCALE_A;
			AH.block<1,8>(7,0) *= SCALE_B;
			AT.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AT.block<3,8>(3,0) *= SCALE_XI_ROT;
			AT.block<1,8>(6,0) *= SCALE_A;
			AT.block<1,8>(7,0) *= SCALE_B;

			adHost[h+t*nFrames] = AH;
			adTarget[h+t*nFrames] = AT;
			adHostF[idx] = AH;
			adTargetF[idx] = AT;
		}
	// Prior가 매우 크다, 따라서 초기에 주어진 파라미터 값을 매우 신뢰한다는 뜻.
	cPrior = VecC::Constant(setting_initialCalibHessian); // setting_initialCalibHessian = 5e9


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	adHostF = new Mat88f[nFrames*nFrames];
	adTargetF = new Mat88f[nFrames*nFrames];

	// Type cast from double to float
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
			adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
		}

	cPriorF = cPrior.cast<float>(); // double to float


	EFAdjointsValid = true;
}



EnergyFunctional::EnergyFunctional()
{
	adHost=0;
	adTarget=0;


	red=0;

	adHostF=0;
	adTargetF=0;
	adHTdeltaF=0;

	nFrames = nResiduals = nPoints = 0;

	HM = MatXX::Zero(CPARS,CPARS); // 4x4 행렬
	bM = VecX::Zero(CPARS); // 4-벡터


	accSSE_top_L = new AccumulatedTopHessianSSE();
	accSSE_top_A = new AccumulatedTopHessianSSE();
	accSSE_bot = new AccumulatedSCHessianSSE();

	resInA = resInL = resInM = 0;
	currentLambda=0;
}
EnergyFunctional::~EnergyFunctional()
{
	for(EFFrame* f : frames)
	{
		for(EFPoint* p : f->points)
		{
			for(EFResidual* r : p->residualsAll)
			{
				r->data->efResidual=0;
				delete r;
			}
			p->data->efPoint=0;
			delete p;
		}
		f->data->efFrame=0;
		delete f;
	}

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	if(adHTdeltaF != 0) delete[] adHTdeltaF;



	delete accSSE_top_L;
	delete accSSE_top_A;
	delete accSSE_bot;
}



// delta = x - x_0 (current - lin.point), 선형화 된 잔차의 $\Delta \bf x$를 계산
void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
{
	// Delta of Pose & photo.params
	if(adHTdeltaF != 0) delete[] adHTdeltaF; // existing dynamic array delete
	// adjoint of Host-Target delta frame?
	adHTdeltaF = new Mat18f[nFrames*nFrames]; // allocate new nFrames**2 1x8 matrix

	// pose delta
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			// Jp * Δp (포즈 p의 투영 위치에 대한 자코비안 Jp과 포즈 증분의 곱 Δp) 항을 미리 계산합니다.
			// 이는 잔차의 선형화된 근사치 r(x) ≈ r(x₀) + JΔx 에서 JΔx의 일부입니다.
			// Δp는 host와 target 프레임의 포즈 증분(Δx_h, Δx_t)에 의해 결정됩니다.
			// adHostF ≈ Jp * J_h, adTargetF ≈ Jp * J_t 이므로,
			// adHTdeltaF^T ≈ (Jp * J_h) * Δx_h + (Jp * J_t) * Δx_t = Jp * Δp 가 됩니다.
			// 이 값을 미리 계산해두면 에너지 계산 시 반복적인 연산을 피할 수 있습니다.
			int idx = h+t*nFrames;

			// Δx_h: frames[h]->data->get_state_minus_stateZero()
			// Δx_t: frames[t]->data->get_state_minus_stateZero()
			// Total derivative를 이용하여 계산 된다. 
			// 상대적인 변화량..
			adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
					        +frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
		}

	// delta of calibration parameters. (current estimate)-(first estimat)
	cDeltaF = HCalib->value_minus_value_zero.cast<float>(); 
	
	// pose와 photo.cal의 delta(=x - x_0)
	// 절대적인 변화량..
	for(EFFrame* f : frames)
	{
		f->delta = f->data->get_state_minus_stateZero().head<8>();
		f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

		for(EFPoint* p : f->points)
			p->deltaF = p->data->idepth-p->data->idepth_zero;
	}

	EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) // HA_top, bA_top,multiThreading
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
				accSSE_top_A, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false,true);
		resInA = accSSE_top_A->nres[0];
	}
	else
	{
		accSSE_top_A->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_A->addPoint<0>(p,this);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false,false);
		resInA = accSSE_top_A->nres[0];
	}
}

// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
				accSSE_top_L, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,true);
		resInL = accSSE_top_L->nres[0];
	}
	else
	{
		accSSE_top_L->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_L->addPoint<1>(p,this);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,false);
		resInL = accSSE_top_L->nres[0];
	}
}





void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
				accSSE_bot, &allPoints, true,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_bot->stitchDoubleMT(red,H,b,this,true);
	}
	else
	{
		accSSE_bot->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_bot->addPoint(p, true);
		accSSE_bot->stitchDoubleMT(red, H, b,this,false);
	}
}

void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
	assert(x.size() == CPARS+nFrames*8);

	VecXf xF = x.cast<float>();
	HCalib->step = - x.head<CPARS>(); // x means incremental. head<n>(): 0번째부터 n-1번째 원소까지 가져옴, minus를 붙임!

	Mat18f* xAd = new Mat18f[nFrames*nFrames]; // 뭘까..? xAd..? Adjoint?
	VecCf cstep = xF.head<CPARS>();            // camera parameter step
	for(EFFrame* h : frames)
	{
		h->data->step.head<8>() = - x.segment<8>(CPARS+8*h->idx); // fh->step.head<8> 호스트 프레임의 업데이트 스텝 가져옴
		h->data->step.tail<2>().setZero(); // 광도 파라미터 업데이트는 0으로 만듦

		for(EFFrame* t : frames) // 타겟 프레임, adHostF*dx_host + adTargetF*dx_target
			xAd[nFrames*h->idx + t->idx] = xF.segment<8>(CPARS+8*h->idx).transpose() *   adHostF[h->idx+nFrames*t->idx]
			            + xF.segment<8>(CPARS+8*t->idx).transpose() * adTargetF[h->idx+nFrames*t->idx];
	}

	if(MT)
		red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
						this, cstep, xAd,  _1, _2, _3, _4), 0, allPoints.size(), 50);
	else
		resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
        const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		EFPoint* p = allPoints[k];

		int ngoodres = 0;
		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
		if(ngoodres==0) // good residual의 개수가 0이면 cont'
		{
			p->data->step = 0;
			continue;
		}
		float b = p->bdSumF;
		b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isActive()) continue;
			b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;
		}

		p->data->step = - b*p->HdiF;
		assert(std::isfinite(p->data->step));
	}
}


double EnergyFunctional::calcMEnergyF()
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta = getStitchedDeltaF();
	// 2 * bM^T * dx + dx^T * HM * dx
	return delta.dot(2*bM + HM*delta); // E=(r_0+J*dx)^2에서 상수 제외한 에너지
}


void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
{

	Accumulator11 E; // scalar accumulator
	E.initialize();
	VecCf dc = cDeltaF; // delta of calibration parameter in float type

	for(int i=min;i<max;i++)
	{
		EFPoint* p = allPoints[i];
		float dd = p->deltaF;

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isLinearized || !r->isActive()) continue;

			Mat18f dp = adHTdeltaF[r->hostIDX+nFrames*r->targetIDX];
			RawResidualJacobian* rJ = r->J; // Jacobian of a residual



			// compute Jp*delta
			float Jp_delta_x_1 =  rJ->Jpdxi[0].dot(dp.head<6>())
						   +rJ->Jpdc[0].dot(dc)
						   +rJ->Jpdd[0]*dd;

			float Jp_delta_y_1 =  rJ->Jpdxi[1].dot(dp.head<6>())
						   +rJ->Jpdc[1].dot(dc)
						   +rJ->Jpdd[1]*dd;

			__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
			__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
			__m128 delta_a = _mm_set1_ps((float)(dp[6]));
			__m128 delta_b = _mm_set1_ps((float)(dp[7]));

			for(int i=0;i+3<patternNum;i+=4)
			{
				// PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
				__m128 Jdelta =            _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx))+i),Jp_delta_x);
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx+1))+i),Jp_delta_y));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF))+i),delta_a));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF+1))+i),delta_b));

				__m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF)+i);
				r0 = _mm_add_ps(r0,r0);
				r0 = _mm_add_ps(r0,Jdelta);
				Jdelta = _mm_mul_ps(Jdelta,r0);
				E.updateSSENoShift(Jdelta);
			}
			for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			{
				float Jdelta = rJ->JIdx[0][i]*Jp_delta_x_1 + rJ->JIdx[1][i]*Jp_delta_y_1 +
								rJ->JabF[0][i]*dp[6] + rJ->JabF[1][i]*dp[7];
				E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2*r->res_toZeroF[i])));
			}
		}
		E.updateSingle(p->deltaF*p->deltaF*p->priorF);
	}
	E.finish();
	(*stats)[0] += E.A;
}




double EnergyFunctional::calcLEnergyF_MT()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	double E = 0;
	for(EFFrame* f : frames)
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);// cwiseProduct: element-wise product(Hadarmard)

	E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

	red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt,
			this, _1, _2, _3, _4), 0, allPoints.size(), 50);

	return E+red->stats[0];
}



EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
	EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);

    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

	nResiduals++;
	r->efResidual = efr;
	return efr;
}
EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
	EFFrame* eff = new EFFrame(fh); // 새로운 EFFrame 동적 할당
	eff->idx = frames.size();       // 인덱스 설정
	frames.push_back(eff);          // EnergyFunctional에 frames 추가

	nFrames++;
	fh->efFrame = eff;              // 프론트엔드에서 백엔드를 가리키도록 설정

	assert(HM.cols() == 8*nFrames+CPARS-8);
	bM.conservativeResize(8*nFrames+CPARS);
	HM.conservativeResize(8*nFrames+CPARS,8*nFrames+CPARS);
	bM.tail<8>().setZero();
	HM.rightCols<8>().setZero();
	HM.bottomRows<8>().setZero();

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	setAdjointsF(Hcalib); // adjoint matrix 계산
	makeIDX();            // 키프레임의 번호 부여, 잔차의 호스트-타겟 인덱스 갱신

	// Initialize connectivity map entries for the new frame.
	// This pre-allocates space for tracking co-observations with other keyframes.
	for(EFFrame* fh2 : frames)
	{
		// eff는 새로이 추가 된 키프레임이다. 
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
		if(fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
	}

	return eff;
}
EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
	//(PointHessian, PointHessian->FrameHessian->EFFrame)
	EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp; // EPPoint "pointed-by"  PointHessian

	EFIndicesValid = false;

	return efp;
}


void EnergyFunctional::dropResidual(EFResidual* r)
{
	EFPoint* p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();


	if(r->isActive())
		r->host->data->shell->statistics_goodResOnThis++;
	else
		r->host->data->shell->statistics_outlierResOnThis++;


    connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
	nResiduals--;
	r->data->efResidual=0;
	delete r;
}
void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int)fh->points.size()==0);
	int ndim = nFrames*8+CPARS-8;// new dimension
	int odim = nFrames*8+CPARS;// old dimension


//	VecX eigenvaluesPre = HM.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//

	// 아래 코드블록은 주변화 할 결과 벡터(b-vector) 원소들을 끝으로 옮기고,
	// Hessian 행렬도도 이에 맞춰 옮긴다.
	if((int)fh->idx != (int)frames.size()-1)
	{
		int io = fh->idx*8+CPARS;	// 현재 프레임 인덱스만큼의 카메라 파라미터 + intrinsics = 포즈(5) 및 광도파라미터(2) + intrinsics(4)가 있다.
		int ntail = 8*(nFrames-fh->idx-1); // 주변화 후 벡터의 차원
		assert((io+8+ntail) == nFrames*8+CPARS);

		// b-vector 계산: 주변화 할 프레임의 상태 변수를 끝으로 옮긴다. (swap)
		Vec8 bTmp = bM.segment<8>(io); // io부터 8개 요소에 대한 벡터
		VecX tailTMP = bM.tail(ntail); // bM의 뒤에서 ntail 개의 요소
		bM.segment(io,ntail) = tailTMP; // bM의 io부터 ntail까지 tailTMP로 대체
		bM.tail<8>() = bTmp;

		// H 행렬에서 주변화 하려는 위치의 열을 끝으로 옮기기(swap)
		MatXX HtmpCol = HM.block(0,io,odim,8); // 0행에서 odim개 행, io열에서 9개 행의 블록을 잡는다.
		MatXX rightColsTmp = HM.rightCols(ntail);
		HM.block(0,io,odim,ntail) = rightColsTmp;
		HM.rightCols(8) = HtmpCol;

		// H 행렬에서 주변화 하려는 위치 행을 끝으로 옮기기 (swap)
		MatXX HtmpRow = HM.block(io,0,8,odim);
		MatXX botRowsTmp = HM.bottomRows(ntail);
		HM.block(io,0,ntail,odim) = botRowsTmp;
		HM.bottomRows(8) = HtmpRow;
	}


//	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<8,8>().diagonal() += fh->prior; // 주변화 할 블록에 prior를 더한다? 왜?
    bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior); // cwise-는 coefficient-wise fh의 prior vector에 Delta prior를 각각 곱함.



//	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";


	// Jacobi Preconditioning step 1
	VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();

	// scale! J... P... step 2
	MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM;

	// invert bottom part!
	Mat88 hpi = HMScaled.bottomRightCorner<8,8>();
	hpi = 0.5f*(hpi+hpi); // to-be symmetric
	hpi = hpi.inverse(); // invserse
	hpi = 0.5f*(hpi+hpi); // to-be symmetric

	// schur-complement!
	MatXX bli = HMScaled.bottomLeftCorner(8,ndim).transpose() * hpi; // H_rm * H_mm^-1
	// noalias()는 메모리 상에 겹치는 부분이 없다는 뜻이다. <= 개발자가 정말로 앨리어싱이 없음을 확신할 때만 써야한다.
	// alias가 존재하는 default의 경우 임시 행렬을 만든다. -> 느려진다.
	HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8,ndim); // H_rr - H_rm * H_mm^-1 * H_mr
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<8>(); // b_r - H_rm * H_mm^-1 * b_m

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	// set.
	HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM = bMScaled.head(ndim);

	// remove from vector, without changing the order!
	for(unsigned int i=fh->idx; i+1<frames.size();i++)
	{
		frames[i] = frames[i+1]; // 한 인덱스씩 위로!
		frames[i]->idx = i; // index reordering
	}
	frames.pop_back();
	nFrames--;
	fh->data->efFrame=0;

	assert((int)frames.size()*8+CPARS == (int)HM.rows());
	assert((int)frames.size()*8+CPARS == (int)HM.cols());
	assert((int)frames.size()*8+CPARS == (int)bM.size());
	assert((int)frames.size() == (int)nFrames);




//	VecX eigenvaluesPost = HM.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	makeIDX();
	delete fh;
}




void EnergyFunctional::marginalizePointsF()
{
	// 0. 사전 조건 확인: 최적화에 필요한 값들이 최신 상태인지 확인한다.
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	// 1. 주변화할 포인트 목록 생성
	// 		- FullSystem::flagPointsForRemoval() 함수에서 PS_MARGINALIZE 플래그가 설정된
	//		  포인트들을 찾는다.
	allPointsToMarg.clear();
	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++) // 프레임의 모든 점을 순회
		{
			EFPoint* p = f->points[i]; // 프레임의 i번째 포인트
			if(p->stateFlag == EFPointStatus::PS_MARGINALIZE) // Point Status가 주변화라면,
			{
				// 1-1. 포인트의 역깊이에 대한 Prior 가중치를 크게 높힌다.
				// 		이는 주변화 과정에서 이 포인트의 깊이 값이 거의 변하지지 않도록 "고정"하는 효과를 준다.
				//		신뢰할 수 있는 깊이 정보를 기반으로 제약 조건을 만들기 위함이다.
				p->priorF *= setting_idepthFixPriorMargFac; // 600 * 600

				// 1-2. 이 포인트가 관측 된 프레임 쌍(host-target)간의 연결성 맵을 업데이트 한다.
				for(EFResidual* r : p->residualsAll) // 포인트에 연결 된 잔차를 순회
					if(r->isActive()) // 활성이며 good이면,
                        connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++; // 엣지에 연결성 추가
			    // 1-3. 주변화 할 목록에 점 추가한다.
				allPointsToMarg.push_back(p); 
			}
		}
	}

	// 2. 주변화할 포인트들의 정보 (Hessian, b-vector)를 계산하기 위한 누적기(accumulator) 초기화.
	accSSE_bot->setZero(nFrames);
	accSSE_top_A->setZero(nFrames);
	
	// 3. 주변화할 각 포인트 대해 정보 추출 및 제거
	for(EFPoint* p : allPointsToMarg)
	{
		// 3-1. 포인트의 정보를 Hessian과 b-vector에 누적한다.
		//		- accSSE_top_A: H_cc, H_cp, b_c 등 카메라와 관련된 부분을 계산한다.
		//		- accSSE_bot  : H_pp, H_pc, b_p 등 포인트와 관련된 부분을 계산한다.
		accSSE_top_A->addPoint<2>(p,this); //주변화 모드로 역깊이 관련 H,J,r 계산
		accSSE_bot->addPoint(p,false); // 최적화 이후 호출 되므로 안에 값 보장, 슈어 보수 관련(H_dd, H_cd)

		// 3-2. 정보 추출이 끝난 포인트는 최적화 시스템(EnergyFunctional)에서 완전히 제거한다.
		removePoint(p);
	}

	// 4. 누적된 정보로부터 전체 H, b 행렬(M, Mb)과 슈어 보수항(Msc, Mbsc)을 조립한다.
	MatXX M, Msc;
	VecX Mb, Mbsc;
	accSSE_top_A->stitchDouble(M,Mb,this,false,false); // stitch가 꾀맨다는 뜻으로, 조립과 상통하는듯.
	accSSE_bot->stitchDouble(Msc,Mbsc,this); // 슈어 보수 행렬 스티칭

	resInM+= accSSE_top_A->nres[0]; // 통계용: 주변화된 잔차 수 업데이트

	// 5. 슈어 보수를 적용하여 포인트 변수를 소거하고, 카메라 변수에 대한 정보만을 남긴다.
	//	  H_cam = H_cc - H_cp * H_pp^-1 * H_pc
	// 주변화 된 Hessian과 b-vector.
	MatXX H =  M-Msc;
    VecX b =  Mb-Mbsc;

	// 6. (Option) 시스템의 게이지 자유도를 제거하기 위해 영공간 직교화 수행
	if(setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) // false
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;

		if(!haveFirstFrame)
			orthogonalize(&b, &H);

	}

	// 7. 최종적으로 계산된 H, b를 시스템의 전역 사전 정보(Prior)인 HM, bM에 더해준다.
	//    setting_margWeightFac은 부정확한 선형화 지점으로 인한 오차를 줄이기 위한 가중치.
	HM += setting_margWeightFac*H;
	bM += setting_margWeightFac*b;

	
	//8. (Option) 전체 사전 정보에 대해 다시 한번 직교화를 수행
	if(setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
		orthogonalize(&bM, &HM);

	// 9. 포인트가 제거되었으므로, 인덱스를 다시 빌드하여 유효성 확보.
	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF()
{


	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_DROP)
			{
				removePoint(p);
				i--;
			}
		}
	}

	EFIndicesValid = false;
	makeIDX();
}


void EnergyFunctional::removePoint(EFPoint* p)
{
	for(EFResidual* r : p->residualsAll)
		dropResidual(r);

	EFFrame* h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->data->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}

void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";


	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
//	if(setting_affineOptModeA <= 0)
//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
//	if(setting_affineOptModeB <= 0)
//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());





	// make Nullspaces matrix
	MatXX N(ns[0].rows(), ns.size());
	for(unsigned int i=0;i<ns.size();i++)
		N.col(i) = ns[i].normalized();



	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	double minSv = 1e10, maxSv = 0;
	for(int i=0;i<SNN.size();i++)
	{
		if(SNN[i] < minSv) minSv = SNN[i];
		if(SNN[i] > maxSv) maxSv = SNN[i];
	}
	for(int i=0;i<SNN.size();i++)
		{ if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

	// N pseudo inverse
	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
	MatXX NNpiT = N*Npi.transpose(); 	// [dim] x [dim].
	MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N' = P_N (projection to N)

	if(b!=0) *b -= NNpiTS * *b;			 // (I-P)b=b-Pb
	if(H!=0) *H -= NNpiTS * *H * NNpiTS; // (I-P)H(I-P) = H-PHP, 따라서 맞음.


//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";

}


void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
{
	if(setting_solverMode & SOLVER_USE_GN) lambda=0;          // false
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5; // false

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	// H = H_active + H_linearized + H_marginalized
	// b = b_active + b_linearized + b_marginalized
	MatXX HL_top, HA_top, H_sc; // H_linearized, H_active, H_schur_complement
	VecX  bL_top, bA_top, bM_top, b_sc; // b_active, b_linearized, b_marginalized

	// 1. 활성(Active) 잔차로부터 H_A와 b_A를 누적합니다. (현재 선형화 지점에서의 정보)
	accumulateAF_MT(HA_top, bA_top,multiThreading);

	// 2. 선형화된(Linearized) 잔차로부터 H_L과 b_L을 누적합니다. (이전 선형화 지점 정보 재사용)
	accumulateLF_MT(HL_top, bL_top,multiThreading);

	// 3. 포인트(depth) 부분을 주변화(marginalize)하기 위한 슈어 보수(Schur Complement) 항을 계산합니다.
	// H_sc = H_cp * H_pp^-1 * H_pc
	// b_sc = H_cp * H_pp^-1 * b_p
	accumulateSCF_MT(H_sc, b_sc,multiThreading);



	// 4. 주변화된(Marginalized) 프레임들의 사전 정보(Prior)를 현재 상태에 맞게 업데이트합니다.
	// b_M = b_prior + H_prior * (x_current - x_linearization)
	bM_top = (bM+ HM * getStitchedDeltaF());







	MatXX HFinal_top;
	VecX bFinal_top;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) // false
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;




		MatXX HT_act =  HL_top + HA_top - H_sc;
		VecX bT_act =   bL_top + bA_top - b_sc;


		if(!haveFirstFrame)
			orthogonalize(&bT_act, &HT_act);

		HFinal_top = HT_act + HM;
		bFinal_top = bT_act + bM_top;





		lastHS = HFinal_top;
		lastbS = bFinal_top;

		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);

	}
	else
	{
		// 모든 헤시안과 b-벡터를 합산하여 카메라 포즈에 대한 전체 시스템을 구성합니다.
		// H_top = H_cc (카메라-카메라 부분)
		HFinal_top = HL_top + HM + HA_top;
		// b_top = b_c - H_cp * H_pp^-1 * b_p
		bFinal_top = bL_top + bM_top + bA_top - b_sc;

		// 로깅을 위해 슈어 보수가 적용되기 전의 H와 b를 저장합니다.
		lastHS = HFinal_top - H_sc;
		lastbS = bFinal_top;

		// LM 알고리즘: H_top의 대각선에 람다를 더하여 damping을 적용합니다.
		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);

		// 최종적으로 슈어 보수 항을 빼서 카메라 포즈에 대한 최종 H 행렬을 완성합니다.
		// H_camera = H_cc - H_cp * (H_pp + lambda*I)^-1 * H_pc
		HFinal_top -= H_sc * (1.0f/(1+lambda));
	}





	// 4. 축소된 선형 시스템 풀이 (카메라 포즈 & 파라미터에 대한 업데이트 스텝 x 계산)
	VecX x;
	if(setting_solverMode & SOLVER_SVD)
	{
		/*************(SVD를 이용한 풀이, 생략)*************/
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++)
		{
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++)
		{
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }

			else Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;

	}
	else
	{
		// 수치적 안정성을 위해 H를 스케일링(Jacobi Preconditioning)한 후,
		// LDLT 분해를 이용하여 Hx=b를 푼다.
		VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
	}



	// 5. Gauge fix
	// 계산 된 업데이트 스텝 x에서 영공간에 해당하는 부분 제거한다.
	// 이를 통해 시스템의 절대적 위치/ 스케일이 발산하는 것을 막는다.
	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
	{
		VecX xOld = x;
		// 업데이트 스텝 'x'를 영공간에 직교하도록 투영하여 게이지를 고정합니다.
		// H 행렬은 "이미 사용되었으므로" 수정할 필요가 없어 nullptr(0)을 전달합니다.
		orthogonalize(&x, nullptr);
	}


	lastX = x; // logging을 위해 촤종 업데이트 스텝 저장


	//resubstituteF(x, HCalib);
	// 6. 역치환(Back-substitution)
	// 계산 된 카메라 업데이트트 스텝 x를 이용하여, 3D 포인트의 역깊이에 대한
	// 업데이트트 스텝을 계산한다.
	currentLambda= lambda;
	resubstituteF_MT(x, HCalib,multiThreading);
	currentLambda=0;


}
void EnergyFunctional::makeIDX()
{
	for(unsigned int idx=0;idx<frames.size();idx++) // 새롭게 키프레임이 추가/제거 되었으므로 인덱스 새로 부여
		frames[idx]->idx = idx;

	allPoints.clear();

	for(EFFrame* f : frames)                     // 모든 활성 키프레임에서
		for(EFPoint* p : f->points)              // EFPoint p로 특정해서,
		{
			allPoints.push_back(p);              // EnergyFunctional의 allPoints에 등록
			for(EFResidual* r : p->residualsAll) // residualsAll을 순회한다.
			{
				r->hostIDX = r->host->idx;       // 위에서 업데이트한 idx로 새로 host-target index 업데이트
				r->targetIDX = r->target->idx;
			}
		}


	EFIndicesValid=true;
}


VecX EnergyFunctional::getStitchedDeltaF() const
{
	VecX d = VecX(CPARS+nFrames*8); d.head<CPARS>() = cDeltaF.cast<double>();
	for(int h=0;h<nFrames;h++) d.segment<8>(CPARS+8*h) = frames[h]->delta;
	return d;
}



}
