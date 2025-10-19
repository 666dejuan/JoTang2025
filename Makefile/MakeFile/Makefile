#指定编译器（定义编辑器为gcc）
CC = gcc

#编译选项（允许使用gdb调试器调试；启用二级优化；开启所有警告；开启额外警告；将警告作为错误处理；添加头文件搜索路径）
CFLAGS = -g -O2 -Wall -Wextra -Werror -I./include


# TODO

# 可执行文件名
TARGET = hello.out

# 源文件列表（找到相应.c文件；也可以使用wildcard自动查找！当前目录！所有.c文件）
SRCS = hello.c lib/add.c lib/mu.c

# 对象文件列表（将.c替换为.o）
OBJS = hello.o lib/add.o lib/mu.o

# 默认目标：生成可执行文件（由目标，依赖和命令组成）
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# 模式规则（告诉make如何从.c文件生成.o文件,为不同目录的文件制定规则）：编译.c文件为.o文件
lib/%.o: lib/%.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理生成的文件（rm表示删除，-f是强制执行；后面两个分别对应.o目标文件和可执行文件）
clean:
	rm -f $(OBJS) $(TARGET)

# 声明伪目标（防止当存在名为 clean 的文件时 make 误判）
.PHONY: clean
