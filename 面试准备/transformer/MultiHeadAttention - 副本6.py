def findTargetSumWays(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """

    """
    01背包
    装满背包的组合（01背包好像没有组合与排列的区别）
    x表示相加数目之和，y表示相减数目之和
    x+y = sum
    x-y = tar
    x = sum+tar/2
    dp[i]背包容量为相加数目和为i，
    dp[4] += dp[4-1]
    dp[0] = 1
    """
    # 需要注意这两个判断条件
    # 当目标和的绝对值大于nums之和时，无论如何都得不到tar，要注意绝对值，不然无法与sum对应
    # 当x不能整除时，也无法凑成
    if abs(target)>sum(nums):
        return 0
    if (target+sum(nums))%2 == 1:
        return 0
    x = (target+sum(nums))//2
    dp = [0]*(x+1)
    dp[0] = 1

    for i in nums:
        for j in range(x,i-1,-1):
            dp[j] += dp[j-i]
    # print(dp)
    return dp[-1]