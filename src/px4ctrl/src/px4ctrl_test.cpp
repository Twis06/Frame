#include <ros/ros.h>
#include "PX4CtrlFSM.h"
// #include "READID.h"
#include <signal.h>
#include <c10/cuda/CUDAStream.h>
// #include <stdlib.h>
// #include <cstring>
// #include <stdio.h>
// #include <sys/types.h>
// #include <sys/socket.h>
// #include <sys/ioctl.h>
// #include <netinet/in.h>
// #include <net/if.h>
// #include <net/if_arp.h>
// #include <arpa/inet.h>
// #include <errno.h>
//kdkd localtiem
// #include <traj_utils/LocalTime.h>
//kdkd toF
// #include <nlink_parser/TofsenseFrame0.h>
//kdkd path
// #include <nav_msgs/Path.h>


// #define ETH_NAME        "l4tbr0"
void mySigintHandler(int sig)
{
    ROS_INFO("[PX4Ctrl] exit...");
    ros::shutdown();
}

int main(int argc, char *argv[])
{

// if(getid()==0){return -1;}
// if(get_mac_id()==0){return -1;}

    ros::init(argc, argv, "px4ctrl");
    ros::NodeHandle nh("~");

    signal(SIGINT, mySigintHandler);  
    ros::Duration(1.0).sleep();

    Parameter_t param;
    param.config_from_ros_handle(nh);

    Controller controller(param);
    NNPolicy policy(param);
    PX4CtrlFSM fsm(param, controller, policy);

    // load model and warmup libtorch
    if (param.mode == "neural")
    {
        ROS_INFO("\033[35m[px4ctrl] Use neural network policy\033[35m.");

        if (!policy.initialize(param.model_path))
        {
            ROS_ERROR("[px4ctrl] Model initialization failed. Shutting down the node.");
            // ros::shutdown();
            // return -1;
        }

        // Warmup the nn before takeoff: the first few rounds of inference may take much longer time
        ROS_INFO("[px4ctrl] The fisrt stage nn warmup start (before takeoff)...");
        for (size_t i = 0; i < 2 * param.ctrl_freq_max; i++)  // 2s inference
        {
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            torch::Tensor state = torch::zeros({13}, torch::kFloat32);
            state[0] = -4.0;
            state[1] = 0.0;
            state[2] = 1.5;
            state = state.unsqueeze(0);  
            state = state.to(policy.device);
            torch::Tensor action = policy.inference(state);  
            action = action.cpu(); 
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            ROS_INFO("[px4ctrl] Warmup: %d/%d, duration=%.4f ms", i, 2 * param.ctrl_freq_max, duration);
        }
        ROS_INFO("\033[32m[px4ctrl] The fisrt stage nn warmup done.\033[32m");
    }
    else
    {
        ROS_INFO("\033[35m[px4ctrl] Use PID controller\033[35m.");
    }


    ros::Rate r(param.ctrl_freq_max);
    // int count = 0;

    torch::Tensor state_gpu = torch::zeros({1, 13}, torch::TensorOptions().device(policy.device).dtype(torch::kFloat32));
    torch::Tensor action_gpu = torch::zeros({1, 4}, torch::TensorOptions().device(policy.device).dtype(torch::kFloat32));
    torch::Tensor state_cpu = torch::zeros({1, 13}, torch::TensorOptions().pinned_memory(true).dtype(torch::kFloat32));
    torch::Tensor action_cpu = torch::zeros({1, 4}, torch::TensorOptions().pinned_memory(true).dtype(torch::kFloat32));
    Eigen::Matrix<float, 13, 1> state_eigen = Eigen::Matrix<float, 13, 1>::Zero();
    Eigen::Matrix<float, 4, 1> action_eigen = Eigen::Matrix<float, 4, 1>::Zero();
    Eigen::Vector3d p;
    p << -4.0, 0.0, 1.5;
    Eigen::Vector3d v;
    v << 0.0, 0.0, 0.0;
    Eigen::Vector3d euler;
    euler << 0.0, 0.0, 0.0;
    Eigen::Vector4d act;
    act << 0.0, 0.0, 0.0, 0.0;

    std::unique_ptr<c10::cuda::CUDAStream> stream;

    torch::NoGradGuard no_grad;

    while (ros::ok())
    {
        r.sleep();
        
        //
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        
        state_eigen.segment<3>(0) = p.cast<float>();
        state_eigen.segment<3>(3) = v.cast<float>();
        state_eigen.segment<3>(6) = euler.cast<float>();
        state_eigen.segment<4>(9) = act.cast<float>();
        state_cpu = torch::from_blob(state_eigen.data(), {1, 13}, torch::kFloat32);
        
        std::chrono::high_resolution_clock::time_point end_0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_0 = end_0 - start;

        //
        std::chrono::high_resolution_clock::time_point start_0 = std::chrono::high_resolution_clock::now();
        
        state_gpu.copy_(state_cpu);
        
        std::chrono::high_resolution_clock::time_point end_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_1 = end_1 - start_0;
        
        //
        std::chrono::high_resolution_clock::time_point start_2 = std::chrono::high_resolution_clock::now();
        
        action_gpu = policy.inference(state_gpu);
        
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_2 = end - start_2;

        //
        std::chrono::high_resolution_clock::time_point start_3 = std::chrono::high_resolution_clock::now();
        
        action_cpu.copy_(action_gpu);
        std::memcpy(action_eigen.data(), action_cpu.data_ptr(), 4 * sizeof(float));
        
        std::chrono::high_resolution_clock::time_point end_3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_3 = end_3 - start_3;    
        
        std::cout << "[px4ctrl] Inference time=" << duration_2.count() << "ms. Send time=" << duration_1.count() 
            << "ms. Convert time=" << duration_3.count() << "ms. Assignment=" << duration_0.count() << "ms\n";
        
        ros::spinOnce();
    }

    return 0;
}
