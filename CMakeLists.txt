CMAKE_MINIMUM_REQUIRED(VERSION 3.25) #cmake最低版本需求，不加入此行会受到警告信息

PROJECT(Sand) #项目名称

INCLUDE_DIRECTORIES(include)

# AUX_SOURCE_DIRECTORY(./src/sand/token SAND_FILES) #把src的sand目录(.)下所有源代码文件和头文件加入变量SAND_FILES
FILE(GLOB_RECURSE SAND_FILES src/sand/*.cpp) # 把src的sand目录(.)下所有源代码文件(递归)和头文件加入变量SAND_FILES

ADD_LIBRARY(libSand ${SAND_FILES}) #将Token编译为一个单独的库

ADD_EXECUTABLE(sand ./src/main.cpp) #生成应用程序 sand

TARGET_LINK_LIBRARIES(sand libSand)
