#pragma once
#include <ros/ros.h>

#define RESET_COLOR     "\033[0m"
#define BLACK           "\033[30m"
#define RED             "\033[31m"
#define GREEN           "\033[32m"
#define YELLOW          "\033[33m"
#define BLUE            "\033[34m"
#define MAGENTA         "\033[35m"
#define CYAN            "\033[36m"
#define WHITE           "\033[37m"
#define BOLD_BLACK      "\033[1m\033[30m"
#define BOLD_RED        "\033[1m\033[31m"
#define BOLD_GREEN      "\033[1m\033[32m"
#define BOLD_YELLOW     "\033[1m\033[33m"
#define BOLD_BLUE       "\033[1m\033[34m"
#define BOLD_MAGENTA    "\033[1m\033[35m"
#define BOLD_CYAN       "\033[1m\033[36m"
#define BOLD_WHITE      "\033[1m\033[37m"

// #define RED "\033[0;1;31m"
// #define GREEN "\033[0;1;32m"
// #define YELLOW "\033[0;1;33m"
// #define BLUE "\033[0;1;34m"
#define PURPLE "\033[0;1;35m"
#define DEEPGREEN "\033[0;1;36m"
// #define WHITE "\033[0;1;37m"
#define RED_IN_WHITE "\033[0;47;31m"
#define GREEN_IN_WHITE "\033[0;47;32m"
#define YELLOW_IN_WHITE "\033[0;47;33m"

#define TAIL "\033[0m"

// font attributes
#define FT_BOLD "\033[1m"
#define FT_UNDERLINE "\033[4m"

// background color
#define BG_BLACK "\033[40m"
#define BG_RED "\033[41m"
#define BG_GREEN "\033[42m"
#define BG_YELLOW "\033[43m"
#define BG_LIGHTBLUE "\033[44m"
#define BG_MAGENTA "\033[45m"
#define BG_BLUE "\033[46m"
#define BG_WHITE "\033[47m"

// font color
#define CL_BLACK(s) "\033[30m" << s << "\033[0m"
#define CL_RED(s) "\033[31m" << s << "\033[0m"
#define CL_GREEN(s) "\033[32m" << s << "\033[0m"
#define CL_YELLOW(s) "\033[33m" << s << "\033[0m"
#define CL_LIGHTBLUE(s) "\033[34m" << s << "\033[0m"
#define CL_MAGENTA(s) "\033[35m" << s << "\033[0m"
#define CL_BLUE(s) "\033[36m" << s << "\033[0m"
#define CL_WHITE(s) "\033[37m" << s << "\033[0m"

#define FG_BLACK "\033[30m"
#define FG_RED "\033[31m"
#define FG_GREEN "\033[32m"
#define FG_YELLOW "\033[33m"
#define FG_LIGHTBLUE "\033[34m"
#define FG_MAGENTA "\033[35m"
#define FG_BLUE "\033[36m"
#define FG_WHITE "\033[37m"

#define FG_NORM "\033[0m"

#define BOX_FUNCTION "["<<__FUNCTION__<<"] "

inline void printFZ() {
  std::cout << GREEN
      "                       _oo0oo_                      \n"
      "                      o8888888o                     \n"
      "                      88\" . \"88                     \n"
      "                      (| -_- |)                     \n"
      "                      0\\  =  /0                     \n"
      "                   ___/‘---’\\___                   \n"
      "                  .' \\|       |/ '.                 \n"
      "                 / \\\\|||  :  |||// \\                \n"
      "                / _||||| -卍-|||||_ \\               \n"
      "               |   | \\\\\\  -  /// |   |              \n"
      "               | \\_|  ''\\---/''  |_/ |              \n"
      "               \\  .-\\__  '-'  ___/-. /              \n"
      "             ___'. .'  /--.--\\  '. .'___            \n"
      "         .\"\" ‘<  ‘.___\\_<|>_/___.’>’ \"\".          \n"
      "       | | :  ‘- \\‘.;‘\\ _ /’;.’/ - ’ : | |        \n"
      "         \\  \\ ‘_.   \\_ __\\ /__ _/   .-’ /  /        \n"
      "    =====‘-.____‘.___ \\_____/___.-’___.-’=====     \n"
      "                       ‘=---=’                      \n"
      "                                                    \n"
      "...Best wishes for you, never fried chicken, never bugs...\n"
      "...........................Amitabha.......................\n"
            << TAIL << std::endl;
}

inline void printKeyboard() {
  std::cout << BLUE
            << "   ┌─────────────────────────────────────────────────────────────┐\n"
               "   │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│\n"
               "   ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\\ │`~ ││\n"
               "   │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│\n"
               "   ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││\n"
               "   │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│\n"
               "   ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│\" '│ Enter  ││\n"
               "   │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│\n"
               "   ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││\n"
               "   │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│\n"
               "   │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │\n"
               "   │      └───┴─────┴───────────────────────┴─────┴───┘          │\n"
               "   └─────────────────────────────────────────────────────────────┘\n"
            << TAIL << std::endl;
}

inline void printCNM() {
  std::cout << PURPLE
            << "  ┏┓　　　┏┓\n"
               "  ┏┛┻━━━┛┻┓\n"
               "  ┃　　　　　　  ┃\n"
               "  ┃　　　━　　　 ┃\n"
               "  ┃　＞　　　＜　┃\n"
               "  ┃　　　　　　　┃\n"
               "  ┃...　⌒　...  ┃\n"
               "  ┃　　　　　　　┃\n"
               "  ┗━┓　　　┏━┛\n"
               "      ┃　　　┃　\n"
               "      ┃　　　┃\n"
               "      ┃　　　┃\n"
               "      ┃　　　┃  Best Wishes\n"
               "      ┃　　　┃  Never BUG!\n"
               "      ┃　　　┃\n"
               "      ┃　　　┗━━━┓\n"
               "      ┃　　　　　　　┣┓\n"
               "      ┃　　　　　　　┏┛\n"
               "      ┗┓┓┏━┳┓┏┛\n"
               "        ┃┫┫　┃┫┫\n"
               "        ┗┻┛　┗┻┛"
            << TAIL << std::endl;
}

#define FAR_INFO_STREAM(x)  { std::cerr << BOX_FUNCTION << "[" << std::fixed << ros::Time::now().toSec() << "] " <<           "[INFO] VINS: "  << x << TAIL << std::endl; }
#define FAR_DEBUG_STREAM(x) { std::cerr << BOX_FUNCTION << "[" << std::fixed << ros::Time::now().toSec() << "] " << GREEN  << "[DEBUG] VINS: " << x << TAIL << std::endl; }
#define FAR_WARN_STREAM(x)  { std::cerr << BOX_FUNCTION << "[" << std::fixed << ros::Time::now().toSec() << "] " << YELLOW << "[WARN] VINS: "  << x << TAIL << std::endl; }
#define FAR_ERROR_STREAM(x) { std::cerr << BOX_FUNCTION << "[" << std::fixed << ros::Time::now().toSec() << "] " << RED    << "[ERROR] VINS: " << x << TAIL << std::endl; }
#define FAR_GREEN_STREAM(x) { std::cerr << BOX_FUNCTION << "[" << std::fixed << ros::Time::now().toSec() << "] " << GREEN  << "[VINS]: "       << x << TAIL << std::endl; }
#define FAR_BLUE_STREAM(x)  { std::cerr << BOX_FUNCTION << "[" << std::fixed << ros::Time::now().toSec() << "] " << BLUE   << "[VINS]: "       << x << TAIL << std::endl; }

# define FAR_WARN_STREAM_THROTTLE(rate, x) \
    do { \
      static double __log_stream_throttle__last_hit__ = 0.0; \
      std::chrono::time_point<std::chrono::system_clock> __log_stream_throttle__now__ = \
      std::chrono::system_clock::now(); \
      if (__log_stream_throttle__last_hit__ + rate <= \
          std::chrono::duration_cast<std::chrono::seconds>( \
          __log_stream_throttle__now__.time_since_epoch()).count()) { \
        __log_stream_throttle__last_hit__ = \
        std::chrono::duration_cast<std::chrono::seconds>( \
        __log_stream_throttle__now__.time_since_epoch()).count(); \
        FAR_WARN_STREAM(x); \
      } \
    } while(0)
