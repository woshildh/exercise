#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

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
void quickSort(vector<int> & nums,int start,int end)
{
	if(start>=end)
	{
		return ;
	}
	int length=nums.size();
	int pos=start,left=start,right=end; //选择第一个作为base
	while(left < right)
	{
		while(right>pos)
		{
			if(nums[right]<nums[pos])
			{
				swap(nums[right],nums[pos]);
				pos=right;
				right=end;
				break;
			}
			else{
				right--;
			}
		}
		while(left<pos)
		{
			if(nums[left]>nums[pos])
			{
				swap(nums[left],nums[pos]);
				pos=left;
				left=start;
				break;
			}
			else{
				left++;
			}
		}
	}
	quickSort(nums,start,pos-1);
	quickSort(nums,pos+1,end);
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
