#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <file.c>"
    exit 1
fi

FILE=$1
if [ ! -f "$FILE" ]; then
    echo "File $FILE does not exist."
    exit 1
fi

BASENAME=$(basename "$FILE" .c)

gcc -o "$BASENAME" "$FILE" -static -fno-stack-protector -fomit-frame-pointer

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

QEMU_BUILD_DIR="/path/to/your/qemu/build"
QEMU_PATH="$QEMU_BUILD_DIR/qemu-x86_64"

if [ ! -x "$QEMU_PATH" ]; then
    echo "QEMU executable not found at $QEMU_PATH."
    exit 1
fi

"$QEMU_PATH" -D "$BASENAME.log" -d plugin -plugin "$QEMU_BUILD_DIR/tests/tcg/plugins/libinsn.so" "$BASENAME"

MAIN_ADDR=$(nm "$BASENAME" | grep ' T main' | awk '{print $1}')
MAIN_ADDR=$(echo "$MAIN_ADDR" | sed 's/^0*//')
MAIN_ADDR="0x$MAIN_ADDR"

MAIN_SIZE=$(nm "$BASENAME" -S | grep ' T main' | awk '{print $2}')
MAIN_SIZE=$(echo "$MAIN_SIZE" | sed 's/^0*//')
MAIN_SIZE="0x$MAIN_SIZE"

echo "Main function address: $MAIN_ADDR"
echo "Main function size: $MAIN_SIZE"

# filter the log file to only include lines related to the main function
sed -i "1,/^$MAIN_ADDR/{/^$MAIN_ADDR/!d;}" "$BASENAME.log"

# for each line in the log file, split by ':', and if the first part is greater
# than MAIN_ADDR + MAIN_SIZE, remove all lines after it this is to filter out
# any lines that are not related to the main function we assume the log file is
# in the format "address: instruction"
awk -F': ' -v addr="$MAIN_ADDR" -v size="$MAIN_SIZE" '
{
    if (strtonum($1) > strtonum(addr) + strtonum(size)) {
        exit
    }
    print $0
}' "$BASENAME.log" > "$BASENAME.filtered.log"
mv "$BASENAME.filtered.log" "$BASENAME.log"

sed -i "s/'//g" "$BASENAME.log"