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

 
#include "util/globalCalib.h"
#include "vector"
 
#include "util/NumType.h"
#include <iostream>
#include <fstream>
#include "util/globalFuncs.h"
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{
class PointHessian;
class FrameHessian;
class CalibHessian;

class EFResidual;


enum ResLocation {ACTIVE=0, LINEARIZED, MARGINALIZED, NONE};
enum ResState {IN=0, OOB, OUTLIER}; // INlier, Out-Of-Bound, OUTLIER

struct FullJacRowT
{
	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
};

class PointFrameResidual
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 메모리의 공간적 정렬을 위해 필요한 매크로

	EFResidual* efResidual;         // backend 최적화에 사용는는 객체와 연결

	static int instanceCounter;     // 전체 잔차 개수를 세기 위한 카운터


	ResState state_state;             // 현재 잔차의 상태 (IN, OOB, OUTLIER)
	double state_energy;              // 현재 상태에서의 에너지(photometric error)
	ResState state_NewState;          // 선형화 과정에서 제안된 새로운 상태
	double state_NewEnergy;           // NewState에서 계산된 에너지 (이상치일 경우 상한값이 적용됨)
	double state_NewEnergyWithOutlier;// NewState에서 계산된, 상한값이 적용되지 않은 순수 에너지


	void setState(ResState s) {state_state = s;} // 상태 설정


	PointHessian* point;	// 잔차에 관여하는 3D 포인트
	FrameHessian* host;		// 포인트가 처음 관측된 키프레임
	FrameHessian* target;	// 포인트가 투영되는 대상 키프레임
	RawResidualJacobian* J; // 상태 변수에 대한 에너지(잔차)의 자코비안


	bool isNew; // 이 잔차가 새로 생성되어 아직 최적화에 포함되지 않았음을 나타내는 플래그


	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT]; // target 프레임에 투영된 8개 패턴의 픽셀 좌표 (subpixel)
	Vec3f centerProjectedTo;						// target 프레임에 투영된 중심점의 (u, v, idepth) 좌표

	~PointFrameResidual();
	PointFrameResidual();
	PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_);
	double linearize(CalibHessian* HCalib); // 자코비안 J와 에너지(잔차)를 계산하는 함수


	void resetOOB()
	{
		state_NewEnergy = state_energy = 0;
		state_NewState = ResState::OUTLIER;

		setState(ResState::IN);
	};
	void applyRes( bool copyJacobians); // linearize()로 계산된 새로운 상태(NewState)를 현재 상태(state_state)로 적용

	void debugPlot();

	void printRows(std::vector<VecX> &v, VecX &r, int nFrames, int nPoints, int M, int res);
};
}
