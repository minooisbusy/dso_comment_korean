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

#include <cmath>

#include <algorithm>

namespace dso
{





// Active Residuals에 대해 에너지와 Jacobian을 계산하여 선형화를 수행 한다.
void FullSystem::linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		/******************01. Linearize a residual(factor)*******************/
		// 모든 잔차 r을 탐색하고 선형화를 수행한다.
		PointFrameResidual* r = activeResiduals[k]; // fullsystem::optimize()에서 선형화 되지 않은 k번째 잔차 를 가져옴
		// 모든 에너지 항은 *status[0]에 누적 된다.
		// 선형화 함수는 최하위 수준의 오차함수의 다양한 Jacobian 계산하고 J에 집어넣는다. 
		// 또한 에너지도 계산하기 때문에 리턴 값으로 energyLeft를 리턴함
		(*stats)[0] += r->linearize(&Hcalib); // Compute sub-blocks of J to combines with other parts

		if(fixLinearization) // 현재 call에서는 false, 최적화가 한 스텝 진행 된 후 호출 된다.
		{
			r->applyRes(true); // d[u]/d[xi]*d[r]/d[idepth]를 계산하고, 상태를 최신화한다.

			if(r->efResidual->isActive()) // isActiveAndGoodNew를 리턴함.
			{
				if(r->isNew)
				{
					PointHessian* p = r->point;
					Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u,p->v, 1);	// projected point assuming infinite depth. KRKip
					Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll*p->idepth_scaled;	// projected point with real depth. KRKip+rho*t
					// parallax on image plane
					float relBS = 0.01*((ptp_inf.head<2>() / ptp_inf[2])-(ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel. 

					
					// Update maxRelBaseline variable with bigger value 
					if(relBS > p->maxRelBaseline) 
						p->maxRelBaseline = relBS;

					p->numGoodResiduals++;
				}
			}
			else
			{
				toRemove[tid].push_back(activeResiduals[k]);
			}
		}
	}
}


void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
		activeResiduals[k]->applyRes(true);
}

// `newFrame`의 에너지 임계치를 현재 잔차 에너지에 따라 동적으로 설정한다.
void FullSystem::setNewFrameEnergyTH()
{

	// collect all residuals and make decision on TH.
	allResVec.clear();
	allResVec.reserve(activeResiduals.size()*2);
	FrameHessian* newFrame = frameHessians.back();

	for(PointFrameResidual* r : activeResiduals)
		if(r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame) // target이 newFrame이고, 에너지(패턴 잔차의합)가 양수인 경우
		{
			allResVec.push_back(r->state_NewEnergyWithOutlier); // 잔차를 하나씩 추가한다.

		}

	if(allResVec.size()==0)
	{
		newFrame->frameEnergyTH = 12*12*patternNum;
		return;		// should never happen, but lets make sure.
	}


	int nthIdx = setting_frameEnergyTHN*allResVec.size(); // 0.7f x 잔차 개수 => 잔차 개수 중 약 70%에서 잔차

	assert(nthIdx < (int)allResVec.size()); // 0에서 1 사이이므로 무조건 작아야함
	assert(setting_frameEnergyTHN < 1);     // 1 이하여야함.

	std::nth_element(allResVec.begin(), allResVec.begin()+nthIdx, allResVec.end()); // 잔차의 위치 찾기
	float nthElement = sqrtf(allResVec[nthIdx]); // 에너지의 제곱근






	//! Maybe this shit is totally hard corded.
    newFrame->frameEnergyTH = nthElement*setting_frameEnergyTHFacMedian; // sqrt(70%energy) * 1.5f
	newFrame->frameEnergyTH = 26.0f*setting_frameEnergyTHConstWeight // 26.0f * 0.5f, 26.0f is experimentally choose.
							  + newFrame->frameEnergyTH*(1-setting_frameEnergyTHConstWeight); // energy * 0.5f => mean or weighted sum
	newFrame->frameEnergyTH = newFrame->frameEnergyTH*newFrame->frameEnergyTH; // square; 에너지는 제곱의 꼴이다.
	newFrame->frameEnergyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight; // squared value(But it's 1.0f)



//
//	int good=0,bad=0;
//	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
//	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
//			meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
//			good, bad);
}
Vec3 FullSystem::linearizeAll(bool fixLinearization)
{
	double lastEnergyP = 0;
	double lastEnergyR = 0;
	double num = 0;

	// 주변화 할 잔차? NUM_THREADS = 6
	std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
	for(int i=0;i<NUM_THREADS;i++) toRemove[i].clear();

	// 아래에서는 직/병렬에 따라 linearize_all_Reductor를 실행는는하느데, fixLinearlization이 false이므로,
	// 앞선 호출에서 채워진 `activeResiduals`에서 하나의 잔차를 선형화 한다.
	if(multiThreading) // 병렬 처리의 경우
	{
		// 원자 함수인 linearizeAll_Reducer를 병렬로 실행함: 활성화 된 잔차의 개수만큼 Jacobian subblocks 계산
		treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
		lastEnergyP = treadReduce.stats[0];
	}
	else // 직렬처리의 경우
	{
		Vec10 stats;
		// min = 0, max = activaResidual.size()
		linearizeAll_Reductor(fixLinearization, toRemove, 0,activeResiduals.size(),&stats,0);
		lastEnergyP = stats[0];
	}


	setNewFrameEnergyTH(); // newFrame의 에너지 임계치를 현재 잔차 에너지에 기반하여 동적으로로 설정한다.


	if(fixLinearization) // 현재 call에서는 false
	{
		// Finalize the state of residuals after optimization is complete.
		// Update the tracking status of each point based on its latest residuals.
		for(PointFrameResidual* r : activeResiduals)
		{
			PointHessian* ph = r->point;
			// Update the state of the last two residuals for each point.
			// This information is used by PointHessian::isOOB to decide if the point should be marginalized.
			if(ph->lastResiduals[0].first == r) // pair; first: PointFrameResidual, second: resState
				ph->lastResiduals[0].second = r->state_state;
			else if(ph->lastResiduals[1].first == r)
				ph->lastResiduals[1].second = r->state_state;
		}

		// Permanently remove residuals that were marked as OUTLIER or OOB.
		// These were collected in the 'toRemove' vector during the final linearization step.
		int nResRemoved=0;
		for(int i=0;i<NUM_THREADS;i++)
		{
			for(PointFrameResidual* r : toRemove[i])
			{
				PointHessian* ph = r->point;
				
				// Clear references to the removed residual from the point's tracking history.
				if(ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].first=0;
				else if(ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].first=0;

				// Find and remove the residual from the point's list of observations.
				for(unsigned int k=0; k<ph->residuals.size();k++)
					if(ph->residuals[k] == r)
					{
						// Drop from the main energy functional and delete the object.
						ef->dropResidual(r->efResidual);
						deleteOut<PointFrameResidual>(ph->residuals,k);
						nResRemoved++;
						break;
					}
			}
		}
	}

	return Vec3(lastEnergyP, lastEnergyR, num); // lastEnergyR은 0인채로 남는다. 활성포인트의 전체 에너지와 개수를 리턴한다.
}




// applies step to linearization point.
bool FullSystem::doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD)
{
//	float meanStepC=0,meanStepP=0,meanStepD=0;
//	meanStepC += Hcalib.step.norm();

	Vec10 pstepfac;
	pstepfac.segment<3>(0).setConstant(stepfacT);
	pstepfac.segment<3>(3).setConstant(stepfacR);
	pstepfac.segment<4>(6).setConstant(stepfacA);


	float sumA=0, sumB=0, sumT=0, sumR=0, sumID=0, numID=0;

	float sumNID=0;

	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
		for(FrameHessian* fh : frameHessians)
		{
			Vec10 step = fh->step;
			step.head<6>() += 0.5f*(fh->step_backup.head<6>());

			fh->setState(fh->state_backup + step);
			sumA += step[6]*step[6];
			sumB += step[7]*step[7];
			sumT += step.segment<3>(0).squaredNorm();
			sumR += step.segment<3>(3).squaredNorm();

			for(PointHessian* ph : fh->pointHessians)
			{
				float step = ph->step+0.5f*(ph->step_backup);
				ph->setIdepth(ph->idepth_backup + step);
				sumID += step*step;
				sumNID += fabsf(ph->idepth_backup);
				numID++;

                ph->setIdepthZero(ph->idepth_backup + step);
			}
		}
	}
	else
	{
		Hcalib.setValue(Hcalib.value_backup + stepfacC*Hcalib.step);
		for(FrameHessian* fh : frameHessians)
		{
			fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
			sumA += fh->step[6]*fh->step[6];
			sumB += fh->step[7]*fh->step[7];
			sumT += fh->step.segment<3>(0).squaredNorm();
			sumR += fh->step.segment<3>(3).squaredNorm();

			for(PointHessian* ph : fh->pointHessians)
			{
				ph->setIdepth(ph->idepth_backup + stepfacD*ph->step);
				sumID += ph->step*ph->step;
				sumNID += fabsf(ph->idepth_backup);
				numID++;

                ph->setIdepthZero(ph->idepth_backup + stepfacD*ph->step);
			}
		}
	}

	sumA /= frameHessians.size();
	sumB /= frameHessians.size();
	sumR /= frameHessians.size();
	sumT /= frameHessians.size();
	sumID /= numID;
	sumNID /= numID;



    if(!setting_debugout_runquiet)
        printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
                sqrtf(sumA) / (0.0005*setting_thOptIterations),
                sqrtf(sumB) / (0.00005*setting_thOptIterations),
                sqrtf(sumR) / (0.00005*setting_thOptIterations),
                sqrtf(sumT)*sumNID / (0.00005*setting_thOptIterations));


	EFDeltaValid=false;
	setPrecalcValues();



	return sqrtf(sumA) < 0.0005*setting_thOptIterations &&
			sqrtf(sumB) < 0.00005*setting_thOptIterations &&
			sqrtf(sumR) < 0.00005*setting_thOptIterations &&
			sqrtf(sumT)*sumNID < 0.00005*setting_thOptIterations;
//
//	printf("mean steps: %f %f %f!\n",
//			meanStepC, meanStepP, meanStepD);
}



// sets linearization point.
void FullSystem::backupState(bool backupLastStep)
{
	if(setting_solverMode & SOLVER_MOMENTUM) // 0x0880 & 0x0200 = 0 -> false
	{
		if(backupLastStep)
		{
			Hcalib.step_backup = Hcalib.step;
			Hcalib.value_backup = Hcalib.value;
			for(FrameHessian* fh : frameHessians)
			{
				fh->step_backup = fh->step;
				fh->state_backup = fh->get_state();
				for(PointHessian* ph : fh->pointHessians)
				{
					ph->idepth_backup = ph->idepth;
					ph->step_backup = ph->step;
				}
			}
		}
		else
		{
			Hcalib.step_backup.setZero();
			Hcalib.value_backup = Hcalib.value;
			for(FrameHessian* fh : frameHessians)
			{
				fh->step_backup.setZero();
				fh->state_backup = fh->get_state();
				for(PointHessian* ph : fh->pointHessians)
				{
					ph->idepth_backup = ph->idepth;
					ph->step_backup=0;
				}
			}
		}
	}
	else
	{
		Hcalib.value_backup = Hcalib.value; // Hcalib을 저장해준다.
		for(FrameHessian* fh : frameHessians)
		{
			fh->state_backup = fh->get_state(); // backup
			for(PointHessian* ph : fh->pointHessians)
				ph->idepth_backup = ph->idepth;
		}
	}
}

// sets linearization point.
void FullSystem::loadSateBackup()
{
	Hcalib.setValue(Hcalib.value_backup);
	for(FrameHessian* fh : frameHessians)
	{
		fh->setState(fh->state_backup);
		for(PointHessian* ph : fh->pointHessians)
		{
			ph->setIdepth(ph->idepth_backup);

            ph->setIdepthZero(ph->idepth_backup);
		}

	}


	EFDeltaValid=false;
	setPrecalcValues();
}

// `calc`ulate `M`arginalized `Energy`
double FullSystem::calcMEnergy()
{
	if(setting_forceAceptStep) return 0;
	// 아래 계산은 뭘까?
	// calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
	//ef->makeIDX();
	//ef->setDeltaF(&Hcalib);
	return ef->calcMEnergyF();

}


void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b)
{
	printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
			res[0],
			sqrtf((float)(res[0] / (patternNum*ef->resInA))),
			ef->resInA,
			ef->resInM,
			a,
			b
	);

}

/********** 슬라이딩 윈도우 안에 있는 키프레임에 GN 최적화**********/
//* makeKeyFrame에서만 호출 된다.
float FullSystem::optimize(int mnumOptIts)
{

	if(frameHessians.size() < 2) return 0;        // 2개 이하면 pair 조차 이룰 수 없다.
	if(frameHessians.size() < 3) mnumOptIts = 20; // 아마도 실험적으로 정한 값
	if(frameHessians.size() < 4) mnumOptIts = 15;






	// get statistics and active residuals.
	// 선형화 되지 않은 잔차를 activeResiduals 컨테이너에 추가함.
	activeResiduals.clear();
	int numPoints = 0;
	int numLRes = 0;

	// 모든 활성 키프레임에 대해 각 프레임에서 선형화 되지 않은 점들을 `activeResiduals`에 추가한다.
	for(FrameHessian* fh : frameHessians)
		for(PointHessian* ph : fh->pointHessians)
		{
			// 모든 잔차를 탐색한다.
			for(PointFrameResidual* r : ph->residuals)
			{
				// r->efResidual에서 선형화 되지 않은 잔차만을 추가함.
				if(!r->efResidual->isLinearized)
				{
					activeResiduals.push_back(r); // 선형화 되지 않은 경우 활성잔차에 추가한다.
					r->resetOOB(); // PointFrameResidual r의 에너지를 0으로, 
								   //state_state를 IN으로, state_Newstate를 Outlier로 설정함.
								   // state_state를 IN으로 설정하는 이유는, 다시 사용 가능한지 보기 위함이다..?
								   //? state_Newstate를 Outlier로 하는이유는? applyRes()에서 바로 사용하지 않기 위해?
				}
				else
					numLRes++; // 선형화 된 잔차 개수 ++
			}
			numPoints++; // 전체 점 ++
		}

    if(!setting_debugout_runquiet)
        printf("OPTIMIZE %d pts, %d active res, %d lin res!\n",ef->nPoints,(int)activeResiduals.size(), numLRes);


	// 모든 활성 잔차 오류와 "야코비안"을 계산. false는 고정된 선형화가 없음을 의미함.
	Vec3 lastEnergy = linearizeAll(false); // "활성" 잔차를 선형화 하고, 예상되는 새로운 상태(정상? 이상치? 영상에 투영되지 않음?)만 지정한다.

	/* setting_forceAcceptStep이 true이므로, 아래 두 함수는 0을 리턴한다.*/
	// 주변화 되어야하는 모든 잔차 에너지의 합
	double lastEnergyL = calcLEnergy(); // calculate Linearized Energy setting_forceAceptStep가 true이기에 0을 리턴함.
	// 주변화 후 남은 사전 에너지(여기서 에너지는 선형화되었고 선형화 지점이 고정되었음을 유의)
	double lastEnergyM = calcMEnergy(); // calculate Marginalized Energy 위와 같이 0을 리턴함




	// 모든 활성 잔차에 대해 Jacobian을 최신화 한다.
	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
	else
		applyRes_Reductor(true,0,activeResiduals.size(),0,0);


    if(!setting_debugout_runquiet)
    {
        printf("Initial Error       \t");
        printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
    }

	debugPlotTracking();



	double lambda = 1e-1;
	float stepsize=1;
	VecX previousX = VecX::Constant(CPARS+ 8*frameHessians.size(), NAN);
	for(int iteration=0;iteration<mnumOptIts;iteration++)
	{
		// solve!
		backupState(iteration!=0); // intriniscs, pose, ab, idepth를 backup
		//solveSystemNew(0);
		solveSystem(iteration, lambda);
		double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
		previousX = ef->lastX;


		if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
		{
			float newStepsize = exp(incDirChange*1.4);
			if(incDirChange<0 && stepsize>1) stepsize=1;

			stepsize = sqrtf(sqrtf(newStepsize*stepsize*stepsize*stepsize));
			if(stepsize > 2) stepsize=2;
			if(stepsize <0.25) stepsize=0.25;
		}

		bool canbreak = doStepFromBackup(stepsize,stepsize,stepsize,stepsize,stepsize);







		// eval new energy!
		Vec3 newEnergy = linearizeAll(false);
		double newEnergyL = calcLEnergy();
		double newEnergyM = calcMEnergy();




        if(!setting_debugout_runquiet)
        {
            printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
				(newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
						lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				incDirChange,
				stepsize);
            printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
        }

		if(setting_forceAceptStep || (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
				lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
		{

			if(multiThreading)
				treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
			else
				applyRes_Reductor(true,0,activeResiduals.size(),0,0);

			lastEnergy = newEnergy;
			lastEnergyL = newEnergyL;
			lastEnergyM = newEnergyM;

			lambda *= 0.25;
		}
		else
		{
			loadSateBackup();
			lastEnergy = linearizeAll(false);
			lastEnergyL = calcLEnergy();
			lastEnergyM = calcMEnergy();
			lambda *= 1e2;
		}


		if(canbreak && iteration >= setting_minOptIterations) break;
	}



	Vec10 newStateZero = Vec10::Zero();
	newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

	frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
			newStateZero);
	EFDeltaValid=false;
	EFAdjointsValid=false;
	ef->setAdjointsF(&Hcalib);
	setPrecalcValues();




	lastEnergy = linearizeAll(true);




	if(!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
    {
        printf("KF Tracking failed: LOST!\n");
		isLost=true;
    }


	statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

	if(calibLog != 0)
	{
		(*calibLog) << Hcalib.value_scaled.transpose() <<
				" " << frameHessians.back()->get_state_scaled().transpose() <<
				" " << sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA))) <<
				" " << ef->resInM << "\n";
		calibLog->flush();
	}

	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		for(FrameHessian* fh : frameHessians)
		{
			fh->shell->camToWorld = fh->PRE_camToWorld;
			fh->shell->aff_g2l = fh->aff_g2l();
		}
	}




	debugPlotTracking();

	return sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

}





void FullSystem::solveSystem(int iteration, double lambda)
{
	ef->lastNullspaces_forLogging = getNullspaces(// 아래 인자들에 저장 된다.
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);

	ef->solveSystemF(iteration, lambda,&Hcalib);
}


// calculate `L`inearized Energy
double FullSystem::calcLEnergy()
{
	if(setting_forceAceptStep) return 0;

	double Ef = ef->calcLEnergyF_MT(); // MT : Multi-Threaded, 내부에서 red->reduce를 사용함.
	return Ef;

}


void FullSystem::removeOutliers()
{
	int numPointsDropped=0;
	for(FrameHessian* fh : frameHessians)
	{
		for(unsigned int i=0;i<fh->pointHessians.size();i++)
		{
			PointHessian* ph = fh->pointHessians[i];
			if(ph==0) continue;

			if(ph->residuals.size() == 0)
			{
				fh->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				fh->pointHessians[i] = fh->pointHessians.back();
				fh->pointHessians.pop_back();
				i--;
				numPointsDropped++;
			}
		}
	}
	ef->dropPointsF();
}




std::vector<VecX> FullSystem::getNullspaces(
		std::vector<VecX> &nullspaces_pose,
		std::vector<VecX> &nullspaces_scale,
		std::vector<VecX> &nullspaces_affA,
		std::vector<VecX> &nullspaces_affB)
{
	// 1. 출력벡ㅌ터들을 초기화 한다.
	nullspaces_pose.clear();
	nullspaces_scale.clear();
	nullspaces_affA.clear();
	nullspaces_affB.clear();


	// 2. 전체 상태 변수의 차원을 계산한다.
	// CPARS: 카메라 내부 파라미터 (4), frameHessian.size()*8: 각 키프레임의 (포즈(6) + 광도(2))
	int n=CPARS+frameHessians.size()*8;
	std::vector<VecX> nullspaces_x0_pre; // 모든 영공간 벡터를 담을 임시 벡터

	// 3. pose에 대해 6개의 기저 벡터 계산
	for(int i=0;i<6;i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		// 모든 활성 키프레임에 대해
		for(FrameHessian* fh : frameHessians)
		{
			// 각 프레임에 대한 지역적인(local) 영공간 성분(fh->nullspaces_pose)을 가져와
			// 전체 상태 벡터의 해당 위치에 "조립(stitch)"한다.
			// fh->nullspaces_pose는 FrameHessian::setStateZero에서 수치 미분으로로 계산된다.
			nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_pose.col(i); // CPAR+fh->idx*8개번재에서부터, 6개,
																				   // 미리 FrameHessian::SetStateZero에서 계산한 포즈 영공간
			// 최적화에 사용된 스케일을 다시 역으로로 적용한다.
			nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;    // 그 여섯 개 중 앞에 3개에 스케일링
			nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;    // 그 여섯 개 중 뒤에 3개에 스케일링
		}
		nullspaces_x0_pre.push_back(nullspace_x0); // 
		nullspaces_pose.push_back(nullspace_x0);   // 
	}
	// 4. Affine 파라미터(a, b)에 대한 2 개의 영공간 기저 벡터 계산
	for(int i=0;i<2;i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for(FrameHessian* fh : frameHessians)
		{
			// 각 프레임의 광도 파라미터(a, b)에 대한 영공간 성분을 조립한다.
			nullspace_x0.segment<2>(CPARS+fh->idx*8+6) = fh->nullspaces_affine.col(i).head<2>();
			nullspace_x0[CPARS+fh->idx*8+6] *= SCALE_A_INVERSE;
			nullspace_x0[CPARS+fh->idx*8+7] *= SCALE_B_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		if(i==0) nullspaces_affA.push_back(nullspace_x0);
		if(i==1) nullspaces_affB.push_back(nullspace_x0);
	}

	// 5. Scale에 대한 1 개의 기저 벡터 계산
	VecX nullspace_x0(n);
	nullspace_x0.setZero();
	// scale의 nullspace
	for(FrameHessian* fh : frameHessians)
	{
		// 각 프레임의 포즈에 대한 스케일 영공간 성분을 조립한다.
		nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_scale;    // translation 변화에 따른 pose(6)의 스케일 영공간
		nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;
		nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
	}
	nullspaces_x0_pre.push_back(nullspace_x0);
	nullspaces_scale.push_back(nullspace_x0);

	return nullspaces_x0_pre;
}

}
