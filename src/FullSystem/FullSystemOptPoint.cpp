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
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace dso
{


/**
 * @brief 활성 키프레임에 투영을 통해 `idepth`만 최적화 한다.
 */
PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
	int nres = 0;
	for(FrameHessian* fh : frameHessians)
	{
		// residual을 point의 target 개수 만큼 초기화
		if(fh != point->host) // fh가 target인 경우이다.
		{
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
			nres++;
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

	bool print = false;//rand()%50==0;

	// 아래 변수는 이제, 한 점에 대해 targets에 대한 잔차를 만들어주는 것이다.
	float lastEnergy = 0;
	float lastHdd=0; // Hdd는 depth에 대한 Hessian. J^2=(w_h*d[r]/d[idepth])^2
	float lastbd=0;  // bd는 depth에 대한 b-vector
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f; // ImmaturePoint에서 역깊이 범위의 중간 값





	// 잔차를 선형화하고, 에너지를 누적한다. `lastHdd`, `lastbd`도 d[f]/d[idepth]의 제곱을 누적한다.
	for(int i=0;i<nres;i++)
	{
		lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth);
		residuals[i].state_state = residuals[i].state_NewState;
		residuals[i].state_energy = residuals[i].state_NewEnergy;
	}

	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
	{
		if(print)
			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
			nres, lastHdd,lastEnergy,currentIdepth);

	// H와 b 그리고 energy를 계산했으니 이제 최적화 계산!
	// 이는 아마도 Gauss-Newton..
	float lambda = 0.1;
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
	{
		float H = lastHdd;
		H *= 1+lambda;
		float step = (1.0/H) * lastbd;
		float newIdepth = currentIdepth - step;

		float newHdd=0; float newbd=0; float newEnergy=0;
		// 한 번 더 에너지를 계산 해 본다.
		for(int i=0;i<nres;i++)
			newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,newHdd, newbd, newIdepth);

		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
		{
			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		// 새로운 에너지가 더 작으면, 업데이트 진행
		if(newEnergy < lastEnergy)
		{
			currentIdepth = newIdepth;
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;
			for(int i=0;i<nres;i++)
			{
				residuals[i].state_state = residuals[i].state_NewState;
				residuals[i].state_energy = residuals[i].state_NewEnergy;
			}

			lambda *= 0.5;
		}
		else// 더 높다면, 더 움직이게 만듦
		{
			lambda *= 5;
		}

		if(fabsf(step) < 0.0001*currentIdepth) // 이동량이 0.01%면 종료
			break;
	}

	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}


	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs)
	{
		if(print) printf("OptPoint: OUTLIER!\n");
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}



	// ImmaturePoint의 idepth를 최적화 하고, 이제 PointHessian으로 만든다.
	// copy values from ImmaturePoint to Point Hessian.
	PointHessian* p = new PointHessian(point, &Hcalib); // point:[ImmaturePoint], p:[PointHessian]
	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));}

	// 초기화!
	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;
	p->setIdepthZero(currentIdepth); // first estimate를 저장
	p->setIdepth(currentIdepth);     // current estimate를 저장
	p->setPointStatus(PointHessian::ACTIVE);

	// 잔차를 residuals에 등록한다. PointFrameResidual이다! 첨본다!
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN)
		{
			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target); 
			r->state_NewEnergy = r->state_energy = 0; // 에너지 초기화
			r->state_NewState = ResState::OUTLIER;    // 보수적인 새로운 상태 초기화
			r->setState(ResState::IN);                // 현재 상태는 INlier로 초기화
			p->residuals.push_back(r);                // 잔차에 추가...

			if(r->target == frameHessians.back())
			{
				p->lastResiduals[0].first = r;
				p->lastResiduals[0].second = ResState::IN;
			}
			else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]))
			{
				p->lastResiduals[1].first = r;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	if(print) printf("point activated!\n");

	statistics_numActivatedPoints++;
	return p;
}



}
