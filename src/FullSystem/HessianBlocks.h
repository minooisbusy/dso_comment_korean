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
#define MAX_ACTIVE_FRAMES 100

 
#include "util/globalCalib.h"
#include "vector"
 
#include <iostream>
#include <fstream>
#include "util/NumType.h"
#include "FullSystem/Residuals.h"
#include "util/ImageAndExposure.h"


namespace dso
{


inline Vec2 affFromTo(const Vec2 &from, const Vec2 &to)	// contains affine parameters as XtoWorld.
{
	return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}


struct FrameHessian;
struct PointHessian;

class ImmaturePoint;
class FrameShell;

class EFFrame;
class EFPoint;

#define SCALE_IDEPTH 1.0f		// scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
#define SCALE_XI_TRANS 0.5f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)


struct FrameFramePrecalc
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	static int instanceCounter;
	FrameHessian* host;	// defines row
	FrameHessian* target;	// defines column

	// precalc values
	Mat33f PRE_RTll;
	Mat33f PRE_KRKiTll;
	Mat33f PRE_RKiTll;
	Mat33f PRE_RTll_0;

	Vec2f PRE_aff_mode;
	float PRE_b0_mode;

	Vec3f PRE_tTll;
	Vec3f PRE_KtTll;
	Vec3f PRE_tTll_0;

	float distanceLL;


    inline ~FrameFramePrecalc() {}
    inline FrameFramePrecalc() {host=target=0;}
	void set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib); // set host to target transformation
};





struct FrameHessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFFrame* efFrame;

	// constant info & pre-calculated values
	//DepthImageWrap* frame;
	FrameShell* shell;

	Eigen::Vector3f* dI;				 // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
	Eigen::Vector3f* dIp[PYR_LEVELS];	 // coarse tracking / coarse initializer. NAN in [0] only.
	float* absSquaredGrad[PYR_LEVELS];  // only used for pixel select (histograms etc.). no NAN.






	int frameID;						// incremental ID for keyframes only!
	static int instanceCounter;
	int idx;

	// Photometric Calibration Stuff
	float frameEnergyTH;	// set dynamically depending on tracking residual
	float ab_exposure;

	bool flaggedForMarginalization;

	std::vector<PointHessian*> pointHessians;				// contains all ACTIVE points.

	// pointHessianMarginalized : 최적화 변수에서 제거되지만, 시스템에 남겨두고 싶은 포린트들을 저장하는 컨테이너
	// * 역깊이를 더이상 최적화 하진 않지만, 포인트가 갖고 있는 정보를 헤시안의 사전정보에 통합(주변화)하여 시스템의 안정성을 유지하는 데 사용
	// * "추적은 불가능하지만(예를 들어, 화면 밖으로 나감), 포인트의 깊이 추정치는 꽤 신뢰할만해서 정보를 버리기 아까울 때" 추가됨
	// * 이후 최적화 과정에서 제약 조건처럼 작용하여, 시스템이 갑자기 불안정해지는 것을 막는다.
	std::vector<PointHessian*> pointHessiansMarginalized;	// contains all MARGINALIZED points (= fully marginalized, usually because point went OOB.)

	// pointHessianOut : Outlier로 판별 되어 최적화에 사용되지 않고 버려지는 포인트들을 저장하는 컨테이너
	// * 물리적으로 불가능한 상태(idepth < 0)
	// * 관측 정보가 전혀 없을 때
	// * 화면 밖으로 벗어나는 등, 추적이 불가능해지거나 포인트 자체의 신뢰도가 낮아 주변화 가치 없을 때
	// * 인라이어로 보기 어려운, 품질이 낮은 포인트
	// 둘은 FullSystem::flagPointsForRemoval()에서 추가 된다.
	std::vector<PointHessian*> pointHessiansOut;		// contains all OUTLIER points (= discarded.).
	std::vector<ImmaturePoint*> immaturePoints;		// contains all OUTLIER points (= discarded.).


	Mat66 nullspaces_pose;
	Mat42 nullspaces_affine;
	Vec6 nullspaces_scale;

	// variable info.
	SE3 worldToCam_evalPT;
	Vec10 state_zero;  // Linearization point
	Vec10 state_scaled; 
	Vec10 state;	// current state, [0-5: worldToCam-leftEps. 6-7: a,b, 8-9: ??]
	Vec10 step;
	Vec10 step_backup;
	Vec10 state_backup;


    EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPT() const {return worldToCam_evalPT;}
    EIGEN_STRONG_INLINE const Vec10 &get_state_zero() const {return state_zero;}
    EIGEN_STRONG_INLINE const Vec10 &get_state() const {return state;}
    EIGEN_STRONG_INLINE const Vec10 &get_state_scaled() const {return state_scaled;}
    EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const {return get_state() - get_state_zero();}


	// precalc values
	SE3 PRE_worldToCam;
	SE3 PRE_camToWorld;
	std::vector<FrameFramePrecalc,Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc;
	MinimalImageB3* debugImage;


    inline Vec6 w2c_leftEps() const {return get_state_scaled().head<6>();}
    inline AffLight aff_g2l() const {return AffLight(get_state_scaled()[6], get_state_scaled()[7]);}
    inline AffLight aff_g2l_0() const {return AffLight(get_state_zero()[6]*SCALE_A, get_state_zero()[7]*SCALE_B);}



	void setStateZero(const Vec10 &state_zero);
	inline void setState(const Vec10 &state)
	{

		this->state = state;
		state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
		state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
		state_scaled[6] = SCALE_A * state[6];
		state_scaled[7] = SCALE_B * state[7];
		state_scaled[8] = SCALE_A * state[8];
		state_scaled[9] = SCALE_B * state[9];

		PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT(); // w2c_leftEps() is incremental, so Pose is updated.
		PRE_camToWorld = PRE_worldToCam.inverse();
		//setCurrentNullspace();
	};
	inline void setStateScaled(const Vec10 &state_scaled)
	{

		this->state_scaled = state_scaled;
		state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
		state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
		state[6] = SCALE_A_INVERSE * state_scaled[6];// makeKeyFrame에서는
		state[7] = SCALE_B_INVERSE * state_scaled[7];// 이 두 개만 들어가겠지
		state[8] = SCALE_A_INVERSE * state_scaled[8];
		state[9] = SCALE_B_INVERSE * state_scaled[9];
		// 만약 w2c_leftEps가 0-vector이면, 항등.
		PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT(); 
		PRE_camToWorld = PRE_worldToCam.inverse();
		// 여기까지 state 변수는 6,7만 값이 있음. 나머지는 0값.
		//setCurrentNullspace();
	};
	inline void setEvalPT(const SE3 &worldToCam_evalPT, const Vec10 &state)
	{

		this->worldToCam_evalPT = worldToCam_evalPT;
		setState(state);
		setStateZero(state);
	};



	// WorldToCam_evalPT를 설정하여 선형화 지점 설정
	// World-to-Host Transformation을 `worldToCam_evalPT`로 삼는다.
	inline void setEvalPT_scaled(const SE3 &worldToCam_evalPT, const AffLight &aff_g2l)
	{
		// The state vector represents the delta from the linearization point.
		// For a new keyframe, the pose delta is zero.
		Vec10 initial_state = Vec10::Zero();
		// [0-5]: Pose delta (se3)
		// [6-7]: Affine brightness delta (a, b)
		// [8-9]: Reserved

		// Set the initial affine parameters from the coarse tracker.
		initial_state[6] = aff_g2l.a;
		initial_state[7] = aff_g2l.b;

		this->worldToCam_evalPT = worldToCam_evalPT;
		setStateScaled(initial_state); // Scale을 벗겨냄
		setStateZero(this->get_state()); // state의 각 값의 nullspace column을 구한다.
	};

	void release();

	inline ~FrameHessian()
	{
		assert(efFrame==0);
		release(); instanceCounter--;
		for(int i=0;i<pyrLevelsUsed;i++)
		{
			delete[] dIp[i];
			delete[]  absSquaredGrad[i];

		}



		if(debugImage != 0) delete debugImage;
	};
	inline FrameHessian()
	{
		instanceCounter++;
		flaggedForMarginalization=false;
		frameID = -1;
		efFrame = 0;
		frameEnergyTH = 8*8*patternNum;



		debugImage=0;
	};


    void makeImages(float* color, CalibHessian* HCalib);

	inline Vec10 getPrior()
	{
		Vec10 p =  Vec10::Zero();
		if(frameID==0)
		{
			p.head<3>() = Vec3::Constant(setting_initialTransPrior);
			p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
			if(setting_solverMode & SOLVER_REMOVE_POSEPRIOR) p.head<6>().setZero();

			p[6] = setting_initialAffAPrior;
			p[7] = setting_initialAffBPrior;
		}
		else
		{
			if(setting_affineOptModeA < 0)
				p[6] = setting_initialAffAPrior;
			else
				p[6] = setting_affineOptModeA;

			if(setting_affineOptModeB < 0)
				p[7] = setting_initialAffBPrior;
			else
				p[7] = setting_affineOptModeB;
		}
		p[8] = setting_initialAffAPrior;
		p[9] = setting_initialAffBPrior;
		return p;
	}


	inline Vec10 getPriorZero()
	{
		return Vec10::Zero();
	}

};

struct CalibHessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	static int instanceCounter;

	VecC value_zero;
	VecC value_scaled;
	VecCf value_scaledf;
	VecCf value_scaledi;
	VecC value;
	VecC step; // It implies that this struct is an optimization variable
	VecC step_backup;
	VecC value_backup;
	VecC value_minus_value_zero; // 

    inline ~CalibHessian() {instanceCounter--;}
	inline CalibHessian()
	{

		VecC initial_value = VecC::Zero();
		initial_value[0] = fxG[0];
		initial_value[1] = fyG[0];
		initial_value[2] = cxG[0];
		initial_value[3] = cyG[0];

		setValueScaled(initial_value);
		value_zero = value; // just first estimate.
		value_minus_value_zero.setZero();

		instanceCounter++;
		for(int i=0;i<256;i++)
			Binv[i] = B[i] = i;		// set gamma function to identity
	};


	// normal mode: use the optimized parameters everywhere!
    inline float& fxl() {return value_scaledf[0];}
    inline float& fyl() {return value_scaledf[1];}
    inline float& cxl() {return value_scaledf[2];}
    inline float& cyl() {return value_scaledf[3];}
    inline float& fxli() {return value_scaledi[0];}
    inline float& fyli() {return value_scaledi[1];}
    inline float& cxli() {return value_scaledi[2];}
    inline float& cyli() {return value_scaledi[3];}



	inline void setValue(const VecC &value)
	{
		// [0-3: Kl, 4-7: Kr, 8-12: l2r]
		this->value = value;
		value_scaled[0] = SCALE_F * value[0]; // fx
		value_scaled[1] = SCALE_F * value[1]; // fy
		value_scaled[2] = SCALE_C * value[2]; // cx
		value_scaled[3] = SCALE_C * value[3]; // cy

		this->value_scaledf = this->value_scaled.cast<float>();
		this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
		this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
		this->value_scaledi[2] = - this->value_scaledf[2] / this->value_scaledf[0];
		this->value_scaledi[3] = - this->value_scaledf[3] / this->value_scaledf[1];
		this->value_minus_value_zero = this->value - this->value_zero;
	};

	inline void setValueScaled(const VecC &value_scaled)
	{
		this->value_scaled = value_scaled;
		this->value_scaledf = this->value_scaled.cast<float>();
		value[0] = SCALE_F_INVERSE * value_scaled[0];
		value[1] = SCALE_F_INVERSE * value_scaled[1];
		value[2] = SCALE_C_INVERSE * value_scaled[2];
		value[3] = SCALE_C_INVERSE * value_scaled[3];

		this->value_minus_value_zero = this->value - this->value_zero;
		this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
		this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
		this->value_scaledi[2] = - this->value_scaledf[2] / this->value_scaledf[0];
		this->value_scaledi[3] = - this->value_scaledf[3] / this->value_scaledf[1];
	};


	float Binv[256];
	float B[256];


	EIGEN_STRONG_INLINE float getBGradOnly(float color)
	{
		int c = color+0.5f;
		if(c<5) c=5;
		if(c>250) c=250;
		return B[c+1]-B[c];
	}

	EIGEN_STRONG_INLINE float getBInvGradOnly(float color)
	{
		int c = color+0.5f;
		if(c<5) c=5;
		if(c>250) c=250;
		return Binv[c+1]-Binv[c];
	}
};


// hessian component associated with one point.
struct PointHessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	static int instanceCounter;
	EFPoint* efPoint;

	// static values
	float color[MAX_RES_PER_POINT];			// colors in host frame
	float weights[MAX_RES_PER_POINT];		// host-weights for respective residuals.



	float u,v;
	int idx;
	float energyTH;
	FrameHessian* host;
	bool hasDepthPrior;

	float my_type;

	float idepth_scaled;
	float idepth_zero_scaled;
	float idepth_zero;
	float idepth;
	float step;
	float step_backup;
	float idepth_backup;

	float nullspaces_scale;
	float idepth_hessian;
	float maxRelBaseline;
	int numGoodResiduals; // PointHessian이 생성 된 후, 이 포인트와 연결 된 Residual이 성공적으로 IN(inlier) 상태로 처리 된 횟수를 누적하는 "카운터"

	enum PtStatus {ACTIVE=0, INACTIVE, OUTLIER, OOB, MARGINALIZED};
	PtStatus status;

    inline void setPointStatus(PtStatus s) {status=s;}


	inline void setIdepth(float idepth) {
		this->idepth = idepth;
		this->idepth_scaled = SCALE_IDEPTH * idepth;
    }
	inline void setIdepthScaled(float idepth_scaled) {
		this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
		this->idepth_scaled = idepth_scaled;
    }
	inline void setIdepthZero(float idepth) {
		idepth_zero = idepth;
		idepth_zero_scaled = SCALE_IDEPTH * idepth;
		nullspaces_scale = -(idepth*1.001 - idepth/1.001)*500;
    }

	// 새로운 키프레임(target)과 point 사이의 관계
	std::vector<PointFrameResidual*> residuals;					// only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
	std::pair<PointFrameResidual*, ResState> lastResiduals[2]; 	// 마지막 두 개(!)에 잔차에 대한 정보를 포함한다. ([0] = 마지막, [1] = 마지막 직전).


	void release();
	PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib);
    inline ~PointHessian() {assert(efPoint==0); release(); instanceCounter--;}


	/** 
	 * @brief 이 포인트가 Out-Of-Bounds 상태인지, 즉 최적화에서 제거되어야 하는지 판단한다.
	 * @param toKeep (사용되지 않음) 유지 될 프레임 목록.
	 * @param toKarg 곧 주변화 될 프레임 목록
	 * @return 제거 대상이면 true, 아니면 false
	*/
	//! Input argument toKeep is never reffered in this function.
	inline bool isOOB(const std::vector<FrameHessian*>& toKeep, const std::vector<FrameHessian*>& toMarg) const
	{
		// 조건 1: 포인트의 관측 정보가 곧 사라질 예정인가?
		int visInToMarg = 0;
		for(PointFrameResidual* r : residuals) // 이 포인트와 연결된 모든 활성 잔차(관측)를 순회
		{
			if(r->state_state != ResState::IN) continue; // 잔차(factor)가 IN(정상)인 경우만 고려
			for(FrameHessian* k : toMarg) 
				if(r->target == k) visInToMarg++;// 관측 대상(target) 프레임이 곧 주변화 될 프레임(toMarg) 목록에 있는지 확인
		}
		
		if((int)residuals.size() >= setting_minGoodActiveResForMarg && // (1) 포인트가 충분히 관측 되었고,
				numGoodResiduals > setting_minGoodResForMarg+10 &&	   // (2) 과거부터 얼마나 꾸준히 좋은지를 나타내는 것(umGoodResiduals)
				(int)residuals.size()-visInToMarg < setting_minGoodActiveResForMarg) // (3) 주변화 이후 남게 될 관측 수가 최소 기준치 미만으로 떨어진다면,
			return true; // 이 포인트는 더이상 유효하지 않으므로 제거 대상으로 판단 (true 반환)
		// 즉, 과거부터 아주 신뢰성 있게 잘 추적되던 좋은 포인트(2)가, 
		// 이번 키프레임 주변화로 인해 앞으로는 관측 수가 너무 부족해져서(3)
		// 더 이상 제역할을 못할 것으로 예상된다면, 차라리 지금 제거하는게 낫다.
			





		// 조건 2: 가장 최근 관측이 OOB 상태인가?
		if(lastResiduals[0].second == ResState::OOB) return true;
		// 조건 3: 포인트가 충분히 관측되지 않는가?
		if(residuals.size() < 2) return false;
		// 조건 4: 최근 두 번의 관측이 모두 Outlier인가?
		if(lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER) return true;

		return false; // 상기 조건을 만족하지 못하면 OOB가 아니다.
	}


	inline bool isInlierNew()
	{
		return (int)residuals.size() >= setting_minGoodActiveResForMarg
                    && numGoodResiduals >= setting_minGoodResForMarg;
	}

};





}
