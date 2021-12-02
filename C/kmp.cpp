#include <stdio.h>
#include <string.h>
#include <memory>

void build_next(char *pattern, int *next, int n);

int kmp_search(char *str, char *pattern) {
    // 求出两个字符串的长度
    int p_len = strlen(pattern), s_len = strlen(str);
    // 计算 next 数组
    int *next = new int[p_len];
    build_next(pattern, next, p_len);
    // 定义两个的位置
    int s_pos = 0, p_pos = 0;
    // 进行遍历
    while(p_pos < p_len && s_pos < s_len) {
        // 如果两个字符串相等
        if(str[s_pos] == pattern[p_pos]) {
            ++p_pos;
            ++s_pos;
        } else if(p_pos  != 0) {
            p_pos = next[p_pos];
        } else {
            s_pos++;
        }
    }
    // printf("p_pos: %d\n", p_pos);
    // 如果搜索到了末尾，则直接返回
    if(p_pos == p_len) {
        return s_pos - p_len;
    }
    return -1;
}

void build_next(char *pattern, int *next, int n) {
    int i = 1, now = 0;
    next[0] = 0;
    while(i < n) {
        if(pattern[i] == pattern[now]) {
            now++;
            next[i] = now;
            i++;
        } else if(now == 0) {
            next[i] = 0;
            i++;
        } else {
            now = next[now - 1];
        }
    }
}

int main() {
    char str[1000];
    char pattern[100];
    while(1) {
        // 接收输入
        scanf("%s", str);
        scanf("%s", pattern);
        int idx = kmp_search(str, pattern);
        printf("Index: %d\n", idx);
    }
    return 0;
}
