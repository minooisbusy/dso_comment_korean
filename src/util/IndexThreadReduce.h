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
#include "util/settings.h"
#include "boost/thread.hpp"
#include <stdio.h>
#include <iostream>



namespace dso
{
using namespace boost::placeholders;

template<typename Running>
class IndexThreadReduce
{

public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	inline IndexThreadReduce()
	{
		// 1. 인덱스 및 스텝 크기 초기화
		nextIndex = 0;
		maxIndex = 0;
		stepSize = 1;
		// 2. ??
		callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);

		running = true;
		// 
		for(int i=0;i<NUM_THREADS;i++)
		{
			isDone[i] = false;
			gotOne[i] = true;
			workerThreads[i] = boost::thread(&IndexThreadReduce::workerLoop, this, i); // `this`는 멤버 함수를 호출하기 때문.
		}

	}
	inline ~IndexThreadReduce()
	{
		running = false;

		exMutex.lock();
		todo_signal.notify_all();
		exMutex.unlock();

		for(int i=0;i<NUM_THREADS;i++)
			workerThreads[i].join();


		printf("destroyed ThreadReduce\n");

	}

	inline void reduce(boost::function<void(int,int,Running*,int)> callPerIndex, int first, int end, int stepSize = 0)
	{

		memset(&stats, 0, sizeof(Running));

//		if(!multiThreading)
//		{
//			callPerIndex(first, end, &stats, 0);
//			return;
//		}


		// stepSize가 0이면, 스레드 개수만큼 일감을 균등하게 나눈다.
		if(stepSize == 0)
			stepSize = ((end-first)+NUM_THREADS-1)/NUM_THREADS; // first=0, end=0 인경우 stepSize는 계속 0??


		//printf("reduce called\n");

		boost::unique_lock<boost::mutex> lock(exMutex); // 수동으로 lock/unlock 할 수 있음.

		// save
		this->callPerIndex = callPerIndex; // 함수 설정
		nextIndex = first;                 
		maxIndex = end;
		this->stepSize = stepSize;

		// go worker threads!
		for(int i=0;i<NUM_THREADS;i++)
		{
			isDone[i] = false;
			gotOne[i] = false;
		}

		// let them start!
		todo_signal.notify_all(); // 스레드들을 깨운다 => 스레드들이 일을 시작한다.


		//printf("reduce waiting for threads to finish\n");
		// 이전 멀티스레딩이 끝날 때까지 기다림.
		// wait for all worker threads to signal they are done.
		while(true)
		{
			// wait for at least one to finish
			done_signal.wait(lock); // unlock 될 때까지 기다린다.
			//printf("thread finished!\n");

			// check if actually all are finished.
			bool allDone = true;
			for(int i=0;i<NUM_THREADS;i++)
				allDone = allDone && isDone[i];

			// all are finished! exit.
			if(allDone)
				break;
		}

		nextIndex = 0;
		maxIndex = 0;
		// Functor를 다시 default으로 되돌림.
		this->callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);

		//printf("reduce done (all threads finished)\n");
	}

	Running stats;

private:
	boost::thread workerThreads[NUM_THREADS];
	bool isDone[NUM_THREADS];
	bool gotOne[NUM_THREADS];

	boost::mutex exMutex;
	boost::condition_variable todo_signal;
	boost::condition_variable done_signal;

	int nextIndex;
	int maxIndex;
	int stepSize;

	bool running;

	boost::function<void(int,int,Running*,int)> callPerIndex;

	void callPerIndexDefault(int i, int j,Running* k, int tid)
	{
		printf("ERROR: should never be called....\n");
		assert(false);
	}

	void workerLoop(int idx)
	{
		boost::unique_lock<boost::mutex> lock(exMutex);

		while(running)
		{
			// try to get something to do.
			int todo = 0;
			bool gotSomething = false;
			if(nextIndex < maxIndex)
			{
				// got something!
				todo = nextIndex;
				nextIndex+=stepSize;
				gotSomething = true;
			}

			// if got something: do it (unlock in the meantime)
			if(gotSomething)
			{
				lock.unlock();

				assert(callPerIndex != 0);

				Running s; memset(&s, 0, sizeof(Running));
				callPerIndex(todo, std::min(todo+stepSize, maxIndex), &s, idx);
				gotOne[idx] = true;
				lock.lock();
				stats += s; // 공유 변수(stats) 접근을 위해 lock 한 것이다.
			}

			// otherwise wait on signal, releasing lock in the meantime.
			else
			{
				if(!gotOne[idx]) // min=0, max=0인 경우 각 스레드에서 한 번씩 실행 됨.
				{
					lock.unlock();
					assert(callPerIndex != 0);
					Running s; memset(&s, 0, sizeof(Running));
					callPerIndex(0, 0, &s, idx); // idx는 스레드 아이디
					gotOne[idx] = true;
					lock.lock();
					stats += s;
				}
				isDone[idx] = true;
				//printf("worker %d waiting..\n", idx);
				done_signal.notify_all();
				todo_signal.wait(lock); // worker는 따로 조건 없이 notify_all() 등이 발생해야 꺠어난다.
			}
		}
	}
};
}
