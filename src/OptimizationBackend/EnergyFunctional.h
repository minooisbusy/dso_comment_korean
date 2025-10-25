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
#include "util/IndexThreadReduce.h"
#include "vector"
#include <math.h>
#include "map"


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
class AccumulatedTopHessian;
class AccumulatedTopHessianSSE;
class AccumulatedSCHessian;
class AccumulatedSCHessianSSE;


extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;



class EnergyFunctional {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	friend class EFFrame;
	friend class EFPoint;
	friend class EFResidual;
	friend class AccumulatedTopHessian;
	friend class AccumulatedTopHessianSSE;
	friend class AccumulatedSCHessian;
	friend class AccumulatedSCHessianSSE;

	EnergyFunctional();
	~EnergyFunctional();

	/**
	 * @brief 새로운 잔차(residual)를 최적화 그래프에 추가합니다.
	 * @param r 추가할 PointFrameResidual 객체 포인터.
	 * @return 생성된 EFResidual 객체 포인터.
	 */
	EFResidual* insertResidual(PointFrameResidual* r);

	/**
	 * @brief 새로운 키프레임을 최적화 그래프에 추가합니다.
	 * @param fh 추가할 FrameHessian 객체 포인터.
	 * @param Hcalib 현재 캘리브레이션 파라미터.
	 * @return 생성된 EFFrame 객체 포인터.
	 */
	EFFrame* insertFrame(FrameHessian* fh, CalibHessian* Hcalib);

	/**
	 * @brief 새로운 3D 포인트를 최적화 그래프에 추가합니다.
	 * @param ph 추가할 PointHessian 객체 포인터.
	 * @return 생성된 EFPoint 객체 포인터.
	 */
	EFPoint* insertPoint(PointHessian* ph);

	/**
	 * @brief 그래프에서 잔차(residual)를 제거합니다.
	 * @param r 제거할 EFResidual 객체 포인터.
	 */
	void dropResidual(EFResidual* r);

	/**
	 * @brief 프레임을 주변화(marginalize)하여 슬라이딩 윈도우에서 제거합니다.
	 * @param fh 주변화할 EFFrame 객체 포인터.
	 */
	void marginalizeFrame(EFFrame* fh);

	/**
	 * @brief 3D 포인트를 그래프에서 완전히 제거합니다.
	 * @param ph 제거할 EFPoint 객체 포인터.
	 */
	void removePoint(EFPoint* ph);



	/**
	 * @brief 특정 조건(PS_MARGINALIZE)을 만족하는 포인트들을 주변화합니다.
	 */
	void marginalizePointsF();

	/**
	 * @brief 특정 조건(PS_DROP)을 만족하는 포인트들을 그래프에서 제거합니다.
	 */
	void dropPointsF();

	/**
	 * @brief Levenberg-Marquardt 알고리즘을 사용하여 최적화 문제를 한 스텝 해결합니다.
	 * @param iteration 현재 최적화 반복 횟수.
	 * @param lambda Levenberg-Marquardt의 람다(damping) 파라미터.
	 * @param HCalib 캘리브레이션 파라미터에 대한 CalibHessian 객체 포인터.
	 */
	void solveSystemF(int iteration, double lambda, CalibHessian* HCalib);

	/**
	 * @brief 주변화된(Marginalized) 부분의 에너지(비용)를 계산합니다.
	 * @return 계산된 에너지 값.
	 */
	double calcMEnergyF();

	/**
	 * @brief 선형화된(Linearized) 부분의 에너지를 계산합니다. (Multi-Threaded)
	 * @return 계산된 에너지 값.
	 */
	double calcLEnergyF_MT();


	/**
	 * @brief 그래프 구조가 변경된 후, 프레임과 잔차의 인덱스를 다시 빌드합니다.
	 */
	void makeIDX();

	/**
	 * @brief 현재 상태와 선형화 지점 간의 차이(delta)를 모든 변수에 대해 미리 계산합니다.
	 * @param HCalib 캘리브레이션 파라미터에 대한 CalibHessian 객체 포인터.
	 */
	void setDeltaF(CalibHessian* HCalib);

	/**
	 * @brief 프레임 간의 Adjoint 행렬을 미리 계산하여 자코비안 계산을 가속화합니다.
	 * @param Hcalib 캘리브레이션 파라미터에 대한 CalibHessian 객체 포인터.
	 */
	void setAdjointsF(CalibHessian* Hcalib);

	// ================== 멤버 변수 ==================
	std::vector<EFFrame*> frames; // 최적화 윈도우 내의 모든 활성 키프레임.
	int nPoints, nFrames, nResiduals; // 현재 그래프에 포함된 포인트, 프레임, 잔차의 총 개수.

	MatXX HM; // 주변화가 완료 된(Marginalization done) 새로운 헤시안 행렬 (Prior 정보).
	VecX bM;  // 주변화가 완료 된(Marginalization done) 새로운 b 벡터 (Prior 정보).

	int resInA, resInL, resInM; // 각각 Active, Linearized, Marginalized 상태에 있는 잔차의 개수.
	MatXX lastHS; // 마지막으로 계산된 전체 헤시안 행렬 (Schur-complement 이전).
	VecX lastbS;  // 마지막으로 계산된 전체 b 벡터 (Schur-complement 이전).
	VecX lastX;   // 마지막 최적화 단계에서 계산된 업데이트 벡터 (dx).

	// Nullspace 관련 벡터들 (게이지 고정용).
	std::vector<VecX> lastNullspaces_forLogging; // 로깅용 Nullspace.
	std::vector<VecX> lastNullspaces_pose;       // Pose(SE3)에 대한 Nullspace.
	std::vector<VecX> lastNullspaces_scale;      // Scale에 대한 Nullspace.
	std::vector<VecX> lastNullspaces_affA;       // Affine 밝기 파라미터 a에 대한 Nullspace.
	std::vector<VecX> lastNullspaces_affB;       // Affine 밝기 파라미터 b에 대한 Nullspace.

	IndexThreadReduce<Vec10>* red; // 멀티스레딩 계산 결과를 취합하기 위한 Reducer 객체.


	/**
	 * @brief 프레임 간의 연결성(공통으로 관측하는 포인트 수)을 저장하는 맵.
	 * 키: (host_id << 32) | target_id
	 * 값: [active residual 수, marginalized residual 수]
	 */
	std::map<uint64_t,
	  Eigen::Vector2i,
	  std::less<uint64_t>,
	  Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>
	  > connectivityMap;

private:
	/**
	 * @brief 모든 변수들의 델타(현재 상태 - 선형화 지점) 값을 하나의 큰 벡터로 합칩니다.
	 * @return 합쳐진 델타 벡터.
	 */
	VecX getStitchedDeltaF() const;

	// 백-치환(Back-substitution) 함수들
	void resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT);
    void resubstituteFPt(const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid);

	// 헤시안 행렬 및 b 벡터 누적 함수들
	void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
	void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
	void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

	void calcLEnergyPt(int min, int max, Vec10* stats, int tid);

	/**
	 * @brief Nullspace에 대해 H와 b를 직교화하여 시스템의 게이지(gauge)를 고정합니다.
	 * @param b b 벡터에 대한 포인터.
	 * @param H 헤시안 행렬에 대한 포인터.
	 */
	void orthogonalize(VecX* b, MatXX* H);

	// 사전 계산된 값들을 저장하는 변수들.
	Mat18f* adHTdeltaF;

	Mat88* adHost;
	Mat88* adTarget;

	Mat88f* adHostF;
	Mat88f* adTargetF;


	VecC cPrior;
	VecCf cDeltaF; // intrinsics delta
	VecCf cPriorF;

	// SSE 최적화를 사용한 헤시안 누적기.
	AccumulatedTopHessianSSE* accSSE_top_L; // for linearized?
	AccumulatedTopHessianSSE* accSSE_top_A; // for Active Residuals?


	AccumulatedSCHessianSSE* accSSE_bot;    // schur complement용?

	std::vector<EFPoint*> allPoints; // 모든 포인트에 대한 포인터 벡터 (반복문 최적화용).
	std::vector<EFPoint*> allPointsToMarg; // 주변화할 포인트에 대한 포인터 벡터.

	float currentLambda; // 현재 Levenberg-Marquardt 람다 값.

};
}
