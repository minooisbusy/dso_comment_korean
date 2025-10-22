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
#include "FullSystem/ImmaturePoint.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace dso
{



// 프레임이 얼마나 오래된지에 따라 주변화 될 프레임인지 판단
void FullSystem::flagFramesForMarginalization(FrameHessian* newFH) //! newFH is NOT USED!
{
	// 1. 고정된 최대 프레임 개수를 초과하는 가장 오래된 프레임들을 주변화 (일반적으로는 사용되지 않는 옵션)
	// setting_minFrameAge(기본값 1) > setting_maxFrames(기본값 7)는 false이므로 이 블록은 거의 실행되지 않습니다.
	if(setting_minFrameAge > setting_maxFrames) // 1 > 7 == false
	{
		// 최대 프레임의 인덱스부터 끝까지 주변화 플래그 True로 만듦
		for(int i=setting_maxFrames;i<(int)frameHessians.size();i++)
		{
			// 0번부터 오래 된 순서이므로, [0, 최대인덱스-setting_maxFrames]까지 flag On!
			FrameHessian* fh = frameHessians[i-setting_maxFrames];
			fh->flaggedForMarginalization = true;
		}
		return;
	}


	int flagged = 0;
	//? 포인트만 주변되고 포즈는 주변화 되지 않는 경우도 있나? -> 없을껄
	//* marginalize all frames that have not enough points.
	for(int i=0;i<(int)frameHessians.size();i++)
	{
		FrameHessian* fh = frameHessians[i]; // one of activated ketframes
		int in = fh->pointHessians.size() + fh->immaturePoints.size(); // # of whole points
		//? 이미 주변화가 진행 된 케이스도 다시 주변화하나? -> 없음
		//? pointHessianOut은 뭐지?
		int out = fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size();


		Vec2 refToFh=AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure,
				frameHessians.back()->aff_g2l(), fh->aff_g2l());

		if( ( // or condition start
			  in < setting_minPointsRemaining *(in+out) || // 키프레임이 관측하는 3차원 포인트가 너무 적은가? (5% 미만)
		      fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow // 새로운 키프레임과 밝기 차이가가 너무 심한가?
		    ) // or condition end
			&& ((int)frameHessians.size())-flagged > setting_minFrames) // 주변화 되는 키프레임을 포함한 전체 키프레임이 최소 키프레임 보다 많도록 유지
		{
//			printf("MARGINALIZE frame %d, as only %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
			fh->flaggedForMarginalization = true; // 위 조건을 만족하면 주변화 하라는 플래그 On!
			flagged++;
		}
		else
		{
//			printf("May Keep frame %d, as %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
		}
	}

	// marginalize one.
	// 최대 키프레임 보다 많은 경우 주변화
	if((int)frameHessians.size()-flagged >= setting_maxFrames)
	{
		double smallestScore = 1;
		FrameHessian* toMarginalize=0; // FrameHessian의 포인터, 아직 할당하지 않음
		FrameHessian* latest = frameHessians.back(); // 가장 최근 키프레임, newFH와 같은 것인듯


		for(FrameHessian* fh : frameHessians)
		{
			//? 안전장치, 프레임 나이(Age)의 차이 보다 크다면?
			// 활성 키프레임 중 하나의 아이디(등록 순서)가 적어도 lastest와 setting_minFrameAg 만큼 떨어져있어야함. 아니면 주변화 X
			// 현재 키프레임이 첫 키프레임이면 주변화 하지 않는다.
			if(fh->frameID > latest->frameID-setting_minFrameAge || fh->frameID == 0) continue;
			//if(fh==frameHessians.front() == 0) continue;

			//*START MARGINALIZATION CONDITION CHECKING*//
			double distScore = 0;
			//* targetPreclac는 키프레임 `fh`와 현재 슬라이딩 윈도우 내의 다른 키프레임간의 기하학적 관계를 미리 캐싱한 저장소
			//* host frame에서 target frame으로 변환을 담고 있다.
			//TODO for loop 해석
			for(FrameFramePrecalc &ffh : fh->targetPrecalc) // fh와 n번째 프레임의 상대 자세
			{
				if(ffh.target->frameID > latest->frameID-setting_minFrameAge+1 || ffh.target == ffh.host) continue;
				distScore += 1/(1e-5+ffh.distanceLL);

			}
			distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);


			if(distScore < smallestScore)
			{
				smallestScore = distScore;
				toMarginalize = fh;
			}
		}

//		printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
//				toMarginalize->frameID, smallestScore);
		toMarginalize->flaggedForMarginalization = true;
		flagged++;
	}

	// [디버깅용] 주변화 플래그가 설정된 후, 윈도우에 남아있는 프레임들의 ID를 출력합니다.
	// printf("FRAMES LEFT (before marginalization): ");
	// for(FrameHessian* fh : frameHessians)
	// 	if(!fh->flaggedForMarginalization) printf("%d ", fh->frameID);
	// printf("\n");
}




void FullSystem::marginalizeFrame(FrameHessian* frame)
{
	// marginalize or remove all this frames points.

	assert((int)frame->pointHessians.size()==0);


	ef->marginalizeFrame(frame->efFrame);

	// drop all observations of existing points in that(marginalized) frame.

	for(FrameHessian* fh : frameHessians)
	{
		if(fh==frame) continue;

		for(PointHessian* ph : fh->pointHessians)
		{
			for(unsigned int i=0;i<ph->residuals.size();i++)
			{
				PointFrameResidual* r = ph->residuals[i];
				if(r->target == frame)
				{
					if(ph->lastResiduals[0].first == r)
						ph->lastResiduals[0].first=0;
					else if(ph->lastResiduals[1].first == r)
						ph->lastResiduals[1].first=0;


					if(r->host->frameID < r->target->frameID)
						statistics_numForceDroppedResFwd++;
					else
						statistics_numForceDroppedResBwd++;

					ef->dropResidual(r->efResidual);
					deleteOut<PointFrameResidual>(ph->residuals,i);
					break;
				}
			}
		}
	}



    {
        std::vector<FrameHessian*> v;
        v.push_back(frame);
        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishKeyframes(v, true, &Hcalib);
    }


	frame->shell->marginalizedAt = frameHessians.back()->shell->id;
	frame->shell->movedByOpt = frame->w2c_leftEps().norm();

	deleteOutOrder<FrameHessian>(frameHessians, frame);
	for(unsigned int i=0;i<frameHessians.size();i++)
		frameHessians[i]->idx = i;




	setPrecalcValues();
	ef->setAdjointsF(&Hcalib);
}




}
