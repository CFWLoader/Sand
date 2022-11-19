CMAKE_MINIMUM_REQUIRED(VERSION 3.25) #cmake最低版本需求，不加入此行会受到警告信息

PROJECT(Sand) #项目名称

AUX_SOURCE_DIRECTORY(./src SRC_LIST) #把当前目录(.)下所有源代码文件和头文件加入变量SRC_LIST

ADD_EXECUTABLE(sand ${SRC_LIST}) #生成应用程序 sand