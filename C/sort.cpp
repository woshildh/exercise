#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
// 冒泡排序
void bubbleSort(vector<int> & nums)
{
	int length=nums.size();
	for(int i=length-1;i>0;i--)
	{
		for(int j=1;j<=i;j++)
		{
			if(nums[j]<nums[j-1])
			{
				swap(nums[j],nums[j-1]);
			}
		}
	}
}
// 归并排序
void mergeSort(vector<int>& nums, vector<int> &temp, int begin, int end) {
	if(begin >= end)
	    return;
	// 递归对左右进行排序
	int size = end - begin + 1;
	int mid = begin + size / 2 - 1;
	mergeSort(nums, temp, begin, mid);
	mergeSort(nums, temp, mid + 1, end);
	// 进行归并
	int ptr = begin, ptr1 = begin, ptr2 = mid + 1;
	while(ptr1 <= mid && ptr2 <= end) {
	    if(nums[ptr1] < nums[ptr2]) {
		temp[ptr++] = nums[ptr1++];
	    } else {
		temp[ptr++] = nums[ptr2++];
	    }
	}
	while(ptr1 <= mid) {
	    temp[ptr++] = nums[ptr1++];
	}
	while(ptr2 <= end) {
	    temp[ptr++] = nums[ptr2++];
	}
	// 将临时数组中的值挪回原来的数组
	for(int i = begin; i <= end; ++i) {
	    nums[i] = temp[i];
	}
}
// 快速排序
void quickSort(vector<int> &nums, int begin, int end) {
	if(begin >= end) {
	    return;
	}
	// 挑选随机数并且交换到前面
	int pivot = rand() % (end - begin + 1);
	swap(nums[begin], nums[begin + pivot]);
	// 取出当前的随机数
	int cur = nums[begin];
	int left = begin, right = end;
	while(left < right) {
	    while(left < right && nums[right] > cur)
		--right;
	    if(left < right) {
		nums[left++] = nums[right];
	    }
	    while(left < right && nums[left] < cur)
		++left;
	    if(left < right) {
		nums[right--] = nums[left];
	    }
	}
	nums[left] = cur;
	quickSort(nums, begin, left - 1);
	quickSort(nums, left + 1, end);
}
// 堆排序
void adjustHeap(vector<int> &nums, int idx, int size) {
	int left = 2 * idx + 1, right = 2 * idx + 2;
	// 求出左右节点最大值的位置
	int max_pos = idx;
	if(left < size && nums[left] > nums[max_pos]) {
	    max_pos = left;
	}
	if(right < size && nums[right] > nums[max_pos]) {
	    max_pos = right;
	}
	// 如果子节点的值大于根节点则进行交换
	if(max_pos != idx) {
	    // 交换根节点和子节点的值
	    swap(nums[max_pos], nums[idx]);
	    // 对子节点重新进行调整
	    adjustHeap(nums, max_pos, size);
	}
}
void heapSort(vector<int> &nums) {
	int size = nums.size();
	if(size <= 1) 
	    return;
	int max_no_leaf = size / 2 - 1;  //最后一个非叶子节点
	// 首先建立堆
	for(int i = max_no_leaf; i >= 0; --i) {
	    adjustHeap(nums, i, size);
	}
	// 调整堆
	for(int i = nums.size() - 1; i > 0; --i) {
	    // 将最大的值放到数组的最后
	    swap(nums[0], nums[i]);
	    // 调整堆
	    adjustHeap(nums, 0, i);
	}
}
int main()
{
	int a[]={19,100,100,10000,4,1,3,2,17,99,10,6,19,30,17};
	vector<int> nums(a,a+15);
	quickSort(nums,0,nums.size()-1);
	//bubbleSort(nums);
	for(int i=0;i<nums.size();i++)
	{
		cout<<nums[i]<<" ";
	}
	cout<<endl;
	return 0;
}
