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


#pragma once

 
#include "util/NumType.h"
#include "vector"
#include <math.h>
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessian;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;






/**
 * @brief 최적화 그래프에서 하나의 factor를 담당하는 factor node
 */
class EFResidual
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	inline EFResidual(PointFrameResidual* org, EFPoint* point_, EFFrame* host_, EFFrame* target_) :
		data(org), point(point_), host(host_), target(target_)
	{
		isLinearized=false;
		isActiveAndIsGoodNEW=false;
		J = new RawResidualJacobian();
		assert(((long)this)%16==0);
		assert(((long)J)%16==0);
	}
	inline ~EFResidual()
	{
		delete J;
	}


	void takeDataF();


	void fixLinearizationF(EnergyFunctional* ef);


	// structural pointers
	PointFrameResidual* data;
	int hostIDX, targetIDX;
	EFPoint* point;
	EFFrame* host;
	EFFrame* target;
	int idxInAll;

	/**
	 * @brief 포즈 $\xi$, 카메라 파라미터 c,역깊이 $\rho$ㄷ대하한 오차의 미분값(Jacobian)을 저장하는 핵심 데이터
	 */
	RawResidualJacobian* J;

	VecNRf res_toZeroF; // 선형화 지점(x_0)에서 잔차값, 선형화 된 잔차의 에너지를 계산할 때 사용
	Vec8f JpJdF;		// J_pose * J_depth에 해당하는 미리 계산된 값


	// status.
	bool isLinearized;

	// if residual is not OOB & not OUTLIER & should be used during accumulations
	bool isActiveAndIsGoodNEW;
	inline const bool &isActive() const {return isActiveAndIsGoodNEW;}
};


enum EFPointStatus {PS_GOOD=0, PS_MARGINALIZE, PS_DROP};

/**
 * @brief 역깊이를 중심으로한 최적화 노드
 */
class EFPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFPoint(PointHessian* d, EFFrame* host_) : data(d),host(host_)
	{
		takeData(); // priorF(depth prior)=0, deltaF(curr-zero) 설정
		stateFlag=EFPointStatus::PS_GOOD;
	}
	void takeData();

	PointHessian* data;


	/**
	 * @brief inverse depth의 prior. 최적화 과정에서 저ㅈ정 포인트의 역깊이가 너무 크게 변하지 않도록 제약을 가한다.
	 * 	      전역 설정으로 현재는 0이다.
	 */
	float priorF;

	/**
	 * @brief Change in inverse depth from the linearization point (idepth - idepth_zero).
	 * set by method of EFPoint takeData()
	 * modified by setDeltaF of EnergyFunctional in EnergyFunctional.cpp:208
	 * read in calcLEngergyPt of EnergyFunctional:360,406, 
	 *         fixLinearizationF() in EnergyFunctionalStructs.cpp:100, 103
	 * 		   addPoint of AccumalatedTopHessianSSE in AccumulatedTopHessian.cpp:47, 54
	 */
	float deltaF;

	// constant info (never changes in-between).
	int idxInPoints;
	EFFrame* host;

	// contains all residuals.
	std::vector<EFResidual*> residualsAll; // 하나의 EFPoint에 대해 host에 대한 target"s"의 residual

	float bdSumF;
	float HdiF;
	float Hdd_accLF;
	VecCf Hcd_accLF;
	float bd_accLF;
	float Hdd_accAF;
	VecCf Hcd_accAF;
	float bd_accAF;


	EFPointStatus stateFlag;
};



class EFFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFFrame(FrameHessian* d) : data(d)
	{
		takeData();
	}
	void takeData();


	Vec8 prior;				// prior hessian (diagonal)
	Vec8 delta_prior;		// = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
	Vec8 delta;				// state - state_zero.



	std::vector<EFPoint*> points;
	FrameHessian* data;
	int idx;	// idx in frames.

	int frameID;
};

}
