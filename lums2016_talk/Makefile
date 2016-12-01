
# A rudimentary build script

.PHONY: clean link_gmb s_analyze c_gcc c_clang



c_gcc: bugs.c
	gcc -Werror -Wall -O2 -o $@ $<

c_clang: bugs.c
	clang-3.5 -Werror -Wall -O2 -o $@ $<

s_analyze: bugs.c
	clang-check-3.5 -analyze $< -- -Werror -Wall -O2

g_sanitizer: bugs.cpp
	g++ -std=c++11 -O1 -g -o $@ $<
c_sanitizer: bugs.cpp
	clang++-3.5 -std=c++11 -O1 -g -fsanitize=address -fno-omit-frame-pointer -o $@ $<

all: c_gcc c_clang s_analyze

clean:
	rm -rf c_gcc c_clang s_analyze g_sanitizer c_sanitizer
