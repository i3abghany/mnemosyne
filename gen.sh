#!/bin/bash

set -e

# Configuration
QEMU_PATH="${QEMU_PATH:-/localhome/mam47/Desktop/qemu}"
QEMU_BUILD_DIR="$QEMU_PATH/build"
QEMU_EXEC_PATH="$QEMU_BUILD_DIR/qemu-x86_64"

BUILD_PLUGIN=false

usage() {
    echo "Usage: $0 [--build-plugin] <file.c>"
    echo "Options:"
    echo "  --build-plugin   Build QEMU and dyntrace plugin if not already built"
    echo "  <file.c>         C source file to compile and run with QEMU"
    exit 1
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-plugin)
                BUILD_PLUGIN=true
                shift
                ;;
            -*)
                echo "Unknown option $1"
                usage
                ;;
            *)
                FILE=$1
                shift
                ;;
        esac
    done

    if [ -z "$FILE" ]; then
        echo "Error: No input file specified."
        usage
    fi
    
    if [ ! -f "$FILE" ]; then
        echo "Error: File $FILE does not exist."
        exit 1
    fi
}

compile_program() {
    local file=$1
    local basename=$(basename "$file" .c)
    
    echo "Compiling $file..." >&2
    gcc -o "$basename" "$file" -static -fno-stack-protector -fomit-frame-pointer
    
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed." >&2
        exit 1
    fi
    
    echo "$basename"
}

check_qemu_executable() {
    if [ ! -x "$QEMU_EXEC_PATH" ]; then
        echo "QEMU executable not found at $QEMU_EXEC_PATH... attempting to build QEMU."
        return 1 
    fi
    return 0
}

build_qemu_and_plugin() {
    echo "Building QEMU and dyntrace plugin..."
    
    local mnemosyne_dir="$(dirname "$(readlink -f "$0")")"
    pushd "$QEMU_PATH" > /dev/null

    if [ ! -f "contrib/plugins/dyntrace.c" ]; then
        echo "Applying dyntrace plugin patch..."
        git apply "$mnemosyne_dir/qemu-dyntrace-plugin.patch"
    fi
    
    mkdir -p build
    cd build

    ../configure --enable-plugins --target-list=x86_64-linux-user
    ninja
    
    echo "QEMU and plugin built successfully."
    popd > /dev/null
}

run_with_qemu() {
    local basename=$1
    
    echo "Running $basename with QEMU and dyntrace plugin..."
    
    set +e
    "$QEMU_EXEC_PATH" -D "$basename.log" -d plugin \
        -plugin "$QEMU_BUILD_DIR/contrib/plugins/libdyntrace.so" "$basename"
    set -e
}

filter_and_clean_log() {
    local basename=$1
    
    local main_addr=$(nm "$basename" | grep ' T main' | awk '{print $1}')
    main_addr=$(echo "$main_addr" | sed 's/^0*//')
    main_addr="0x$main_addr"

    local main_size=$(nm "$basename" -S | grep ' T main' | awk '{print $2}')
    main_size=$(echo "$main_size" | sed 's/^0*//')
    main_size="0x$main_size"

    echo "Main function address: $main_addr"
    echo "Main function size: $main_size"

    # Filter the log file to only include lines related to the main function
    sed -i "1,/^$main_addr/{/^$main_addr/!d;}" "$basename.log"

    # For each line in the log file, split by the first ',', and if the first
    # part is greater than main_addr + main_size, remove all lines after it
    awk -F': ' -v addr="$main_addr" -v size="$main_size" '
    {
        if (strtonum($1) > strtonum(addr) + strtonum(size)) {
            exit
        }
        print $0
    }' "$basename.log" > "$basename.filtered.log"
    mv "$basename.filtered.log" "$basename.log"

    # Remove all single quotes and format the log
    sed -i "s/'//g" "$basename.log"

    # Substitute the first comma with a colon to make splitting easier as
    # instructions may contain commas
    sed -i 's/\([0-9a-fA-Fx]*\), \(.*\)/\1: \2/' "$basename.log"
    
    echo "Log filtering and cleaning complete."
}

main() {
    parse_arguments "$@"
    
    local basename=$(compile_program "$FILE")
    
    if ! check_qemu_executable || [ "$BUILD_PLUGIN" = true ]; then
        build_qemu_and_plugin
    fi
    
    run_with_qemu "$basename"
    
    filter_and_clean_log "$basename"
    
    echo "Processing complete. Filtered log saved to $basename.log"
}

main "$@"