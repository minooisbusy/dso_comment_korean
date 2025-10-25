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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
int PointFrameResidual::instanceCounter = 0;


long runningResID=0;


PointFrameResidual::PointFrameResidual(){assert(false); instanceCounter++;}

PointFrameResidual::~PointFrameResidual(){assert(efResidual==0); instanceCounter--; delete J;}

PointFrameResidual::PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_) :
	point(point_),
	host(host_),
	target(target_)
{
	efResidual=0;
	instanceCounter++;
	resetOOB();
	J = new RawResidualJacobian();
	assert(((long)J)%16==0);

	isNew=true;
}



// Back-end 최적화에 사용되는 미분 준비
double PointFrameResidual::linearize(CalibHessian* HCalib)
{
	state_NewEnergyWithOutlier=-1;

	if(state_state == ResState::OOB) // Out-of-Bound이면, 새로운 상태도 OOB로 설정하고, state_energy를 그대로 리턴
		{ state_NewState = ResState::OOB; return state_energy; }

	// 이 잔차에 해당하는 사전에 계산 된 값들을 가져온다.
	// 값들은 KRKi, 현재 값에서 변환, 초기화 지점에서 변환 등을 포함한다.
	//! adjoint는 등장하지 않음. 
	//! 현재 계산은 -> 연쇄법칙의 local Jacobian
	//! adHostF는 -> 연쇄법칙의 Global transformation matrix
	FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);
	float energyLeft=0;
	const Eigen::Vector3f* dIl = target->dI; // target의 intensity, dx, dy 포인터 획득
	//const float* const Il = target->I;
	
	// 사전에 계산 된 값들(precalc)에서 기하학적인 값들 가져옴
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
	const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;

	// 포인트의 밝기값
	const float * const color = point->color;
	// 포인트의 가중치, 아마도 수식 7번
	const float * const weights = point->weights;

	// 사전에 계산 된 값들(precalc)에서 (상대적) 광도 파라미터 갖고옴
	Vec2f affLL = precalc->PRE_aff_mode;
	float b0 = precalc->PRE_b0_mode;


	Vec6f d_xi_x, d_xi_y;
	Vec4f d_C_x, d_C_y;
	float d_d_x, d_d_y;
	{
		float drescale, u, v, new_idepth; // u,v 는 normalized coordinate
		float Ku, Kv;
		Vec3f KliP;

		// 보간을 고려한 이미지 영역 내부에 있는지 확인
		if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
				PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{ state_NewState = ResState::OOB; return state_energy; }

		centerProjectedTo = Vec3f(Ku, Kv, new_idepth); // 


		// diff d_idepth, 깊이 값에 대해 u,v를 미분한 결과
		d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*SCALE_IDEPTH*HCalib->fxl();
		d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*SCALE_IDEPTH*HCalib->fyl();




		// diff calib, intrinsics에 대해 u,v를 미분한 결과
		d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));
		d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) * HCalib->fyli();
		d_C_x[0] = KliP[0]*d_C_x[2];
		d_C_x[1] = KliP[1]*d_C_x[3];

		d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) * HCalib->fxli();
		d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
		d_C_y[0] = KliP[0]*d_C_y[2];
		d_C_y[1] = KliP[1]*d_C_y[3];

		// 스케일링
		d_C_x[0] = (d_C_x[0]+u)*SCALE_F;
		d_C_x[1] *= SCALE_F;
		d_C_x[2] = (d_C_x[2]+1)*SCALE_C;
		d_C_x[3] *= SCALE_C;

		d_C_y[0] *= SCALE_F;
		d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
		d_C_y[2] *= SCALE_C;
		d_C_y[3] = (d_C_y[3]+1)*SCALE_C;


		// pose twist에 대해 u,v 미분 결과
		d_xi_x[0] = new_idepth*HCalib->fxl();
		d_xi_x[1] = 0;
		d_xi_x[2] = -new_idepth*u*HCalib->fxl();
		d_xi_x[3] = -u*v*HCalib->fxl();
		d_xi_x[4] = (1+u*u)*HCalib->fxl();
		d_xi_x[5] = -v*HCalib->fxl();

		d_xi_y[0] = 0;
		d_xi_y[1] = new_idepth*HCalib->fyl();
		d_xi_y[2] = -new_idepth*v*HCalib->fyl();
		d_xi_y[3] = -(1+v*v)*HCalib->fyl();
		d_xi_y[4] = u*v*HCalib->fyl();
		d_xi_y[5] = u*HCalib->fyl();
	}


	// 결과를 RawResidualJacobian*에 저장. 예를 들어 Jpdxi는 point를 xi로 미분.
	// J_p_d_* 형식. Jacobian point derivatived by *
	{
		// d[r]/d[xi]
		J->Jpdxi[0] = d_xi_x;
		J->Jpdxi[1] = d_xi_y;

		// d[r]/d[c]
		J->Jpdc[0] = d_C_x;
		J->Jpdc[1] = d_C_y;

		// d[r]/d[rho]
		J->Jpdd[0] = d_d_x;
		J->Jpdd[1] = d_d_y;

	}






	float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
	float JabJIdx_00=0, JabJIdx_01=0, JabJIdx_10=0, JabJIdx_11=0;
	float JabJab_00=0, JabJab_01=0, JabJab_11=0;

	float wJI2_sum = 0;

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
		// 패턴 위치에서 투영이 영상 경계 내에 있는지 확인
		if(!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{ state_NewState = ResState::OOB; return state_energy; }

		projectedTo[idx][0] = Ku;
		projectedTo[idx][1] = Kv;


        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);


		// residual을 파라미터 a로 미분
		float drdA = (color[idx]-b0); // alpha(host color - beta)가 원본이다.
		if(!std::isfinite((float)hitColor[0]))
		{ state_NewState = ResState::OOB; return state_energy; }

		// 수식 7번의 가중치
		float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
		// 
        w = 0.5f*(w + weights[idx]);



		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += w*w*hw *residual*residual*(2-hw);

		{
			if(hw < 1) hw = sqrtf(hw);
			hw = hw*w;

			hitColor[1]*=hw;
			hitColor[2]*=hw;

			J->resF[idx] = residual*hw;

			J->JIdx[0][idx] = hitColor[1];
			J->JIdx[1][idx] = hitColor[2];
			J->JabF[0][idx] = drdA*hw;
			J->JabF[1][idx] = hw;

			JIdxJIdx_00+=hitColor[1]*hitColor[1];
			JIdxJIdx_11+=hitColor[2]*hitColor[2];
			JIdxJIdx_10+=hitColor[1]*hitColor[2];

			JabJIdx_00+= drdA*hw * hitColor[1];
			JabJIdx_01+= drdA*hw * hitColor[2];
			JabJIdx_10+= hw * hitColor[1];
			JabJIdx_11+= hw * hitColor[2];

			JabJab_00+= drdA*drdA*hw*hw;
			JabJab_01+= drdA*hw*hw;
			JabJab_11+= hw*hw;


			wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]); // some criteria..

			if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;
			if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;

		}
	}

	J->JIdx2(0,0) = JIdxJIdx_00;
	J->JIdx2(0,1) = JIdxJIdx_10;
	J->JIdx2(1,0) = JIdxJIdx_10;
	J->JIdx2(1,1) = JIdxJIdx_11;
	J->JabJIdx(0,0) = JabJIdx_00;
	J->JabJIdx(0,1) = JabJIdx_01;
	J->JabJIdx(1,0) = JabJIdx_10;
	J->JabJIdx(1,1) = JabJIdx_11;
	J->Jab2(0,0) = JabJab_00;
	J->Jab2(0,1) = JabJab_01;
	J->Jab2(1,0) = JabJab_01;
	J->Jab2(1,1) = JabJab_11;

	state_NewEnergyWithOutlier = energyLeft;

	if(energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2)
	{
		energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
		state_NewState = ResState::OUTLIER;
	}
	else
	{
		state_NewState = ResState::IN;
	}

	state_NewEnergy = energyLeft;
	return energyLeft;
}



void PointFrameResidual::debugPlot()
{
	if(state_state==ResState::OOB) return;
	Vec3b cT = Vec3b(0,0,0);

	if(freeDebugParam5==0)
	{
		float rT = 20*sqrt(state_energy/9);
		if(rT<0) rT=0; if(rT>255)rT=255;
		cT = Vec3b(0,255-rT,rT);
	}
	else
	{
		if(state_state == ResState::IN) cT = Vec3b(255,0,0);
		else if(state_state == ResState::OOB) cT = Vec3b(255,255,0);
		else if(state_state == ResState::OUTLIER) cT = Vec3b(0,0,255);
		else cT = Vec3b(255,255,255);
	}

	for(int i=0;i<patternNum;i++)
	{
		if((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0]-3 && projectedTo[i][1] < hG[0]-3 ))
			target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1],cT);
	}
}



/**
 * @brief linearize()로 계산된 새로운 상태(NewState)를 현재 상태(state_state)로 적용
 */
void PointFrameResidual::applyRes(bool copyJacobians)
{
	if(copyJacobians)
	{
		if(state_state == ResState::OOB)
		{
			assert(!efResidual->isActiveAndIsGoodNEW);
			return;	// can never go back from OOB
		}
		if(state_NewState == ResState::IN)// Inlier인 경우,
		{
			efResidual->isActiveAndIsGoodNEW=true;
			efResidual->takeDataF(); // 데이터 스왑 후(의미 모름), [u]/d[xi]와 d[r]/d[idepth]의 곱셈을 구한다.
		}
		else
		{
			efResidual->isActiveAndIsGoodNEW=false; // OUTLIER의 경우일 것이다.
		}
	}

	setState(state_NewState); // state_NewState를 현재 스테이트로 적용함(apply)
	state_energy = state_NewEnergy;
}
}
