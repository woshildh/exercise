#include <stdio.h>
#include <stdlib.h>

// 定义跳跃表的最大层数为32层
const int MAX_LEVEL = 32;

typedef struct node {
    int value;
    int key;
    struct node *next[0];   //后继指针数组，柔性数组可以实现结构体的边长
} Node;

typedef struct skip_list {  //跳跃表结构
    int level;   // 跳跃表目前的层数
    Node *head;   // 指向头结点
} skip_list;

int randomLevel() {
    int level = 1;
    while(rand() % 2 && level < MAX_LEVEL)
        ++level;
    return level;
}

Node *create_node(int level, int key, int value) {
    Node *sl_node = (Node*)malloc(sizeof(Node) + MAX_LEVEL * sizeof(Node*));
    if(sl_node == NULL) {
        printf("Create Node failed.\n");
        return NULL;
    }
    sl_node->key = key;
    sl_node->value = value;
    for(int i = 0; i < MAX_LEVEL; ++i) {
        // printf("i: %d, ", i);
        sl_node->next[i] = NULL;
    }
    return sl_node;
}

skip_list *create_skip_list() {
    skip_list *sl = (skip_list*)malloc(sizeof(skip_list));
    if(sl == NULL) {
        printf("malloc failed.\n");
        return NULL;
    }
    sl->level = 0;
    Node *node = create_node(0, -1, -1);
    sl->head = node; // 这个节点是无效的，不能被查找到
    return sl;
}

void free_skip_list(skip_list *sl) {
    if(sl == NULL) {
        return;
    }
    // 先逐个释放节点
    Node *q = NULL, *p = sl->head;
    while(p) {
        q = p->next[0];
        free(p);
        p = q;
    }
    // 释放sl本身
    free(sl);
}

void print_skip_list(skip_list *sl) {
    if(sl == NULL) {
        return;
    }
    // 先逐个释放节点
    Node *q = NULL, *p = sl->head->next[0];
    printf("Skip list: ");
    while(p) {
        printf("(%d, %d), ", p->key, p->value);
        q = p->next[0];
        p = q;
    }
    printf(".\n");
}

/**
* 1. 查找到在每层的插入位置，更新 update 数组 
* 2. 随机产生一个层数
* 3. 从高层向下插入，与普通链表的插入完全相同
*/
bool insert(skip_list *sl, int key, int value) {
    Node *update[MAX_LEVEL];
    // printf("sl: %d\n", sl);
    Node *q = NULL, *p = sl->head;
    int i = sl->level - 1;   //目前跳表的层数
    // printf("i: %d\n", i);
    // 从最高层往下找需要插入的位置，并更新 update
    for(; i >= 0; --i) {
        // printf("i: %d\n", i);
        while(p && (q = p->next[i]) && q && q->key < key) {
            p = q;
        }
        // 记录下当前这一层，大于等于Key的节点的指针
        update[i] = p;
    }
    // printf("=========\n");
    // 如果节点q和key的值相同则直接设置value并返回
    if(q && q->key == key) {
        q->value = value;
        return true;
    }
    // 产生一个随机的层数
    int level = randomLevel();
    // 如果新生成的层数比跳表的层数大就需要更新跳表
    // 目标就是确定新增加的层的第一个节点的起始位置
    if(level > sl->level) {
        // 在 update 数组中将新添加的层指向 header
        for(i = sl->level; i < level; ++i) {
            update[i] = sl->head;
        }
        sl->level = level;
    }
    // printf("level: %d\n", level);
    // 新建一个待插入的节点，一层一层插入
    q = create_node(level, key, value);
    // printf("Key: %d, value: %d\n", q->key, q->value);
    if(q == NULL) {
        return false;
    }
    // printf("value: %d\n", q->value);
    // 逐层更新节点的指针，和普通链表插入一样
    for(i = level - 1; i >= 0; --i) {
        if(update[i] &&  q) {
            q->next[i] = update[i]->next[i];
            update[i]->next[i] = q;
        }
    }
    return true;
}

/**
* 删除节点
*/
bool erase(skip_list *sl, int key) {
    Node *update[MAX_LEVEL];
    Node *q = NULL, *p = sl->head;
    int i = sl->level - 1;
    for(; i >= 0; --i) {
        while((q = p->next[i]) && q->key < key) {
            p = q;
        }
        update[i] = p;
    }
    // 如果q的key不等于待删除的key，则直接返回
    if(!q || q->key != key) {
        return false;
    }
    // 逐层删除
    for(i = sl->level - 1; i >= 0; --i) {
        if(update[i]->next[i] == q) {
            update[i]->next[i] = q->next[i];
        }
        // 如果删除的节点是最高层节点则 level--
        if(sl->head->next[i] == NULL) {
            sl->level--;
        }
    }
    // 释放节点所占用的内存
    free(q);
    q = NULL;
    return true;
}

int* search(skip_list *sl, int key) {
    Node *q = NULL, *p = sl->head;
    int i = sl->level - 1;
    for(; i >= 0; --i) {
        while(p && (q = p->next[i]) && q->key < key) {
            p = q;
        }
        if(q && key == q->key) {
            return &(q->value);
        }
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    skip_list *sl = create_skip_list();
    printf("Create skip list succeed. %d\n", sl);
    // 插入元素
    int nums[5][2] = {{2, 5}, {6, 8}, {10, 100}, {3, 78}, {456, 234}};
    for(int i = 0; i < 5; ++i) {
        bool state = insert(sl, nums[i][0], nums[i][1]);
        // 插入元素
        printf("Insert (%d, %d): %d\n", nums[i][0], nums[i][1], state);
    }
    printf("------------split-----------------\n");
    // 打印跳表
    print_skip_list(sl);
    printf("------------split-----------------\n");
    // 查找跳表
    for(int i = 0; i < 5; ++i) {
        // 查询元素
        int *result = search(sl, nums[i][0]);
        if(result == NULL) {
            printf("Search %d failed.\n", nums[i][0]);
        } else {
            printf("Search %d result is %d.\n", nums[i][0], *result);
        }
    }
    printf("------------split-----------------\n");
    // 删除几个元素
    int del_nums[3] = {6, 2, 7};
    for(int i = 0; i < 3; ++i) {
        // 删除元素
        bool state = erase(sl, del_nums[i]);
        // 元素
        printf("Delete %d: %d\n", del_nums[i], state);        
    }
    printf("------------split-----------------\n");
    // 查找跳表
    for(int i = 0; i < 5; ++i) {
        // 查询元素
        int *result = search(sl, nums[i][0]);
        if(result == NULL) {
            printf("Search %d failed.\n", nums[i][0]);
        } else {
            printf("Search %d result is %d.\n", nums[i][0], *result);
        }
    }    
    // 释放跳表
    free_skip_list(sl);
    sl = NULL;
    return 0;
}
