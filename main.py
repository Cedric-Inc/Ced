
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left = 0
        right = len(s)-1

        while left<=right:
            if not s[left].isalnum():
                left+=1
                continue
            if not s[right].isalnum():
                right-=1
                continue

            if s[left] == s[right]:
                left+=1
                right-=1
            else:
                return False
        return True


s = Solution()

print(s.isPalindrome("A man, a plan, a canal: Panama"))