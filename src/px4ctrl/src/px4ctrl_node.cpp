#include <ros/ros.h>
#include "PX4CtrlFSM.h"
// #include "READID.h"
#include <signal.h>
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
        ROS_INFO("\033[35m[px4ctrl] Use neural network policy!\033[35m.");

        if (!policy.initialize(param.model_path))
            ROS_ERROR("[px4ctrl] Model initialization failed. Shutting down the node.");

        // Warmup the nn before takeoff: the first few rounds of inference may take much longer time
        ROS_INFO("[px4ctrl] The fisrt stage nn warmup start (before takeoff)...");
        for (size_t i = 0; i < 2 * param.ctrl_freq_max; i++)  // 2s inference
            policy.warmup_step(true);
        ROS_INFO("\033[32m[px4ctrl] The fisrt stage nn warmup done.\033[32m");
    }
    else
        ROS_INFO("\033[35m[px4ctrl] Use PID controller!\033[35m.");

    //kdkd local_time print info
    // fsm.local_time_data.my_id = param.drone_id;
    // fsm.local_time_data.formation_nums = param.formation_num;
    //--------------------------

    ros::Subscriber state_sub =
        nh.subscribe<mavros_msgs::State>("/mavros/state",
                                         10,
                                         boost::bind(&State_Data_t::feed, &fsm.state_data, _1));

    ros::Subscriber extended_state_sub =
        nh.subscribe<mavros_msgs::ExtendedState>("/mavros/extended_state",
                                                 10,
                                                 boost::bind(&ExtendedState_Data_t::feed, &fsm.extended_state_data, _1));

    ros::Subscriber odom_sub =
        nh.subscribe<nav_msgs::Odometry>("odom",
                                         100,
                                         boost::bind(&Odom_Data_t::feed, &fsm.odom_data, _1),
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());

    ros::Subscriber cmd_sub =
        nh.subscribe<quadrotor_msgs::PositionCommand>("cmd",
                                                      100,
                                                      boost::bind(&Command_Data_t::feed, &fsm.cmd_data, _1),
                                                      ros::VoidConstPtr(),
                                                      ros::TransportHints().tcpNoDelay());
    
    ros::Subscriber ctrl_FCU_sub = 
        nh.subscribe<mavros_msgs::AttitudeTarget>("/hil_node/fcu_ctrl",  // Note: receive cmd for HIL experiments
                                                 1,  // TODO: check if 1 is suitable
                                                 boost::bind(&Controller_Output_t::feed, &fsm.ctrl_FCU_data, _1),
                                                 ros::VoidConstPtr(),
                                                 ros::TransportHints().tcpNoDelay());

    ros::Subscriber imu_sub =
        nh.subscribe<sensor_msgs::Imu>("/mavros/imu/data", // Note: do NOT change it to /mavros/imu/data_raw !!!
                                       100,
                                       boost::bind(&Imu_Data_t::feed, &fsm.imu_data, _1),
                                       ros::VoidConstPtr(),
                                       ros::TransportHints().tcpNoDelay());

    ros::Subscriber rc_sub;
    if (!param.takeoff_land.no_RC) // mavros will still publish wrong rc messages although no RC is connected
    {
        rc_sub = nh.subscribe<mavros_msgs::RCIn>("/mavros/rc/in",
                                                 10,
                                                 boost::bind(&RC_Data_t::feed, &fsm.rc_data, _1));
    }

    ros::Subscriber bat_sub =
        nh.subscribe<sensor_msgs::BatteryState>("/mavros/battery",
                                                100,
                                                boost::bind(&Battery_Data_t::feed, &fsm.bat_data, _1),
                                                ros::VoidConstPtr(),
                                                ros::TransportHints().tcpNoDelay());

    ros::Subscriber takeoff_land_sub =
        nh.subscribe<quadrotor_msgs::TakeoffLand>("takeoff_land",
                                                  100,
                                                  boost::bind(&Takeoff_Land_Data_t::feed, &fsm.takeoff_land_data, _1),
                                                  ros::VoidConstPtr(),
                                                  ros::TransportHints().tcpNoDelay());

    ros::Subscriber if_switch2hover_sub = 
         nh.subscribe<std_msgs::Empty>("/attitude_recovery",
                                    1,
                                    boost::bind(&Switch2Hover::feed, &fsm.if_switch2hover, _1),
                                    ros::VoidConstPtr(),
                                    ros::TransportHints().tcpNoDelay());

    ros::Subscriber fsm_start_sub = 
    nh.subscribe<std_msgs::Empty>("/px4ctrl/fsm_start",
                                1,
                                boost::bind(&Switch2Hover::feed_false, &fsm.if_switch2hover, _1),
                                ros::VoidConstPtr(),
                                ros::TransportHints().tcpNoDelay());
                           

    fsm.ctrl_FCU_pub = nh.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 1);
    fsm.traj_start_trigger_pub = nh.advertise<geometry_msgs::PoseStamped>("/traj_start_trigger", 10);

    fsm.debug_pub = nh.advertise<quadrotor_msgs::Px4ctrlDebug>("/debugPx4ctrl", 10); // debug

    fsm.set_FCU_mode_srv = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    fsm.arming_client_srv = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    fsm.reboot_FCU_srv = nh.serviceClient<mavros_msgs::CommandLong>("/mavros/cmd/command");

    //kdkd time check
    // fsm.local_time_pub = nh.advertise<traj_utils::LocalTime>("/localTime_from_px4",10);
    
    // ros::Subscriber other_local_times_sub =
    //         nh.subscribe<traj_utils::LocalTime>("/broadcast_local_time_to_other",
    //                                             100,
    //                                             boost::bind(&local_time_Date_t::feed, &fsm.local_time_data, _1),
    //                                             ros::VoidConstPtr(),
    //                                             ros::TransportHints().tcpNoDelay());


    // //kdkd node state
    // fsm.px4_state_to_drone_node_pub = nh.advertise<traj_utils::to_drone_state>("/node_state",10);

    // //kdkd  optical_flow path
    // fsm.pub_optical_flow_path = nh.advertise<nav_msgs::Path>("/optical_path", 1000);    
    // fsm.pub_optical_odometry = nh.advertise<nav_msgs::Odometry>("/optical_odom", 1000);  
                                                               
    ros::Duration(0.5).sleep();

    if (param.takeoff_land.no_RC)
    {
        ROS_WARN("PX4CTRL] Remote controller disabled, be careful!");
    }
    else
    {
        ROS_INFO("PX4CTRL] Waiting for RC");
        while (ros::ok())
        {
            ros::spinOnce();
            if (fsm.rc_is_received(ros::Time::now()))
            {
                ROS_INFO("[PX4CTRL] RC received.");
                break;
            }
            ros::Duration(0.1).sleep();
        }
    }

    int trials = 0;
    while (ros::ok() && !fsm.state_data.current_state.connected)
    {
        ros::spinOnce();
        ros::Duration(1.0).sleep();
        if (trials++ > 5)
            ROS_ERROR("Unable to connnect to PX4!!!");
    }

    if (param.spin_mode == 0) 
    {
        printf("param.spin_mode 0");

        ros::Rate r(param.ctrl_freq_max);
        // int count = 0;

        while (ros::ok())
        {
            r.sleep();

            // if ( count ++ == 50 )
            // {
            //     //kdkd localtime msg
            //     ros::Time t_start = ros::Time::now();
            //     traj_utils::LocalTime timeMsg;
            //     timeMsg.start_time = t_start;
            //     timeMsg.drone_id = param.drone_id;
            //     timeMsg.no_syc = fsm.local_time_data.not_synchronized;
            //     fsm.local_time_pub.publish(timeMsg);
            //     //--------------------
            //     count = 0;
            // }
            ros::spinOnce();
            ros::Time t_start_process = ros::Time::now();
            fsm.process(); // We DO NOT rely on feedback as trigger, since there is no significant performance difference through our test.
            ros::Time t_end_process = ros::Time::now();
            if ((t_end_process - t_start_process).toSec() > 1/param.ctrl_freq_max)
            {
                std::cout << "\033[1;32m time of fsm.process >:1/400" << std::endl;
            }
        }
    }
    else 
    {
        printf("param.spin_mode 1");

        ros::Rate r(1.0 / param.spin_interval);
        int max_interval = round(param.control_interval / param.spin_interval);
        int spin_count = 0;
        bool trigger_moment = false;

        while (ros::ok())
        {
            r.sleep();
            ros::spinOnce();

            if (param.mode == "neural" && param.policy_modality == 2 && !trigger_moment && fsm.hil_cmd_is_received(ros::Time::now()))  
            // when in HIL experiment, run the control at the first moment the HIL node publish the cmd
            {
                spin_count = 0;
                fsm.process();
                trigger_moment = true;
            }
            else if (!spin_count)
                fsm.process();
                
            if (++spin_count >= max_interval)
                spin_count = 0;
        }
    }

    return 0;
}
