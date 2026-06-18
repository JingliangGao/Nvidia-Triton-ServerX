#!/bin/bash

# 设置依赖仓库绝对路径
onnx_include="<path-to-onnx-include>"     # eg, "/home/kylin/onnxruntime/include/onnxruntime/core/session"
onnx_lib="<path-to-onnx-lib>"             # eg, "/home/kylin/onnxruntime/build/Linux/RelWithDebInfo"
llama_include="<path-to-llama-include>"   # eg, "/home/kylin/llamacpp_backend/third_party/include"
llama_lib="<path-to-llama-lib>"           # eg, "/home/kylin/llamacpp_backend/third_party/lib"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 判断是否为sudoers
if [ "$EUID" -ne 0 ]; then
    SUDO_ER='sudo'
else
    SUDO_ER=''
fi

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}



# 显示菜单
show_menu() {
    echo "========================================"
    echo "   Triton Inference Server 编译脚本"
    echo "========================================"
    echo ""
    echo "请选择要编译的角色："
    echo ""
    echo "1) Server (服务端)"
    echo "2) Client (基础客户端)"
    echo "3) ONNX Runtime Backend"
    echo "4) Llama.cpp Backend"
    echo "5) Python Backend"
    echo "6) 测试环境 (Tests)"
    echo "7) 全部编译 (All)"
    echo "8) 编译debian包 (debian)"
    echo "0) 退出"
    echo ""
    echo -n "请输入选择 [0-8]: "
}

# 执行编译
run_cmake() {
    local build_dir=$1
    local cmake_args=$2
    
    # 创建构建目录
    if [ ! -d "$build_dir" ]; then
        mkdir -p "$build_dir"
        print_info "创建构建目录: $build_dir"
    fi
    
    cd "$build_dir" || exit 1
    
    print_info "开始执行 CMake 配置..."
    print_info "CMake 命令: cmake $cmake_args .."
    
    # 执行 cmake
    if eval "cmake $cmake_args .."; then
        print_success "CMake 配置成功！"
        print_info "进入构建目录: cd $build_dir"
        print_info "执行编译: make -j\$(nproc)"
        print_info "安装: make install"
    else
        print_error "CMake 配置失败！"
        exit 1
    fi
    
    cd - > /dev/null || exit 1
}

# 安装依赖
install_dependencies() {
    print_info "安装依赖..."
    # 在这里添加安装依赖的命令
    ${SUDO_ER} apt-get update
    ${SUDO_ER} apt-get install -y \
        debhelper cmake  rapidjson-dev  libprotobuf-dev libgrpc++-dev \
        protobuf-compiler-grpc libgtest-dev libb64-dev python3-grpc-tools \
        pybind11-dev libarchive-dev lsb-release libglib2.0-dev nlohmann-json3-dev libboost-all-dev
    #   kytensor-llm kytensor-llm-dev  prometheus-cpp-dev  libboost1.83-all-dev  debhelper-compat
}

# 编译 Server
build_server() {
    print_info "编译 Server (服务端)..."
    local cmake_args="-DCMAKE_INSTALL_PREFIX=\`pwd\`/install -DTRITON_ENABLE_SERVER=ON -DTRITON_ENABLE_CLIENT=OFF -DTRITON_ENABLE_ONNX_BACKEND=OFF -DTRITON_ENABLE_GPU=OFF "
    run_cmake "build_server" "$cmake_args"
}

# 编译基础客户端
build_client() {
    print_info "编译 Client (基础客户端)..."
    local cmake_args="-DCMAKE_INSTALL_PREFIX=\`pwd\`/install -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=ON -DTRITON_ENABLE_ONNX_BACKEND=OFF -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_ENABLE_TESTS=ON -DTRITON_ENABLE_PYTHON_HTTP=ON -DTRITON_ENABLE_PYTHON_GRPC=ON"
    run_cmake "build_client" "$cmake_args"
}

# 编译 ONNX Runtime Backend
build_onnx() {
    print_info "编译 ONNX Runtime Backend..."
    
    # 检查路径是否存在
    if [ ! -d "$onnx_include" ]; then
        print_warning "ONNX Runtime 包含路径不存在: $onnx_include"
        echo -n "请输入正确的 ONNX Runtime 包含路径: "
        read -r onnx_include
    fi
    
    if [ ! -d "$onnx_lib" ]; then
        print_warning "ONNX Runtime 库路径不存在: $onnx_lib"
        echo -n "请输入正确的 ONNX Runtime 库路径: "
        read -r onnx_lib
    fi
    
    local cmake_args="-DCMAKE_INSTALL_PREFIX=\`pwd\`/install -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=OFF -DTRITON_ENABLE_ONNX_BACKEND=ON -DTRITON_ONNXRUNTIME_INCLUDE_PATHS=$onnx_include -DTRITON_ONNXRUNTIME_LIB_PATHS=$onnx_lib"
    run_cmake "build_onnx" "$cmake_args"
}

# 编译 Llama.cpp Backend
build_llamacpp() {
    print_info "编译 Llama.cpp Backend..."
    
    # 检查路径是否存在
    if [ ! -d "$llama_include" ]; then
        print_warning "Llama.cpp 包含路径不存在: $llama_include"
        echo -n "请输入正确的 Llama.cpp 包含路径: "
        read -r llama_include
    fi
    
    if [ ! -d "$llama_lib" ]; then
        print_warning "Llama.cpp 库路径不存在: $llama_lib"
        echo -n "请输入正确的 Llama.cpp 库路径: "
        read -r llama_lib
    fi
    
    local cmake_args="-DCMAKE_INSTALL_PREFIX=\`pwd\`/install -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=OFF -DTRITON_ENABLE_ONNX_BACKEND=OFF -DTRITON_ENABLE_LLAMACPP_BACKEND=ON -DTRITON_LLAMACPP_INCLUDE_PATHS=$llama_include -DTRITON_LLAMACPP_LIB_PATHS=$llama_lib"
    run_cmake "build_llamacpp" "$cmake_args"
}

# 编译 Python Backend
build_python() {
    print_info "编译 Python Backend..."
    local cmake_args="-DCMAKE_INSTALL_PREFIX=\`pwd\`/install -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=OFF -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_PYTHON_BACKEND=ON"
    run_cmake "build_python" "$cmake_args"
}

# 编译测试环境
build_tests() {
    print_info "编译测试环境..."
    local cmake_args="-DCMAKE_INSTALL_PREFIX=\`pwd\`/install -DTRITON_ENABLE_SERVER=OFF -DTRITON_ENABLE_CLIENT=ON -DTRITON_ENABLE_ONNX_BACKEND=OFF -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_EXAMPLES=OFF -DTRITON_ENABLE_TESTS=ON -DTRITON_ENABLE_PYTHON_HTTP=ON -DTRITON_ENABLE_PYTHON_GRPC=ON -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_METRICS_GPU=OFF"
    run_cmake "build_tests" "$cmake_args"
}

# 全部编译
build_all() {
    print_info "开始全部编译..."
    build_server
    build_client
    build_onnx
    build_llamacpp
    build_python
    build_tests
    print_success "所有编译完成！"
}

build_debian() {
    print_info "编译 Debian 包..."
    dpkg-buildpackage -T clean
    dpkg-buildpackage -us -uc -b 
}

# 主函数
main() {
    # 检查是否在正确的目录
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "当前目录不存在 CMakeLists.txt 文件！"
        print_error "请在 Triton 源代码根目录执行此脚本。"
        exit 1
    fi

    # 安装依赖
    install_dependencies
    
    while true; do
        show_menu
        read -r choice
        
        case $choice in
            1)
                build_server
                break
                ;;
            2)
                build_client
                break
                ;;
            3)
                build_onnx
                break
                ;;
            4)
                build_llamacpp
                break
                ;;
            5)
                build_python
                break
                ;;
            6)
                build_tests
                break
                ;;
            7)
                build_all
                break
                ;;
            8)
                build_debian
                break
                ;;
            0)
                print_info "退出脚本。"
                exit 0
                ;;
            *)
                print_error "无效的选择，请重新输入！"
                ;;
        esac
    done
    
    echo ""
    print_success "编译配置完成！"
    echo ""
    echo "后续步骤(build_debian越过以下步骤)："
    echo "1. 进入对应的构建目录: cd build_*"
    echo "2. 执行编译: make -j\$(nproc)"
    echo "3. 安装: make install"
    echo ""
}

# 执行主函数
main