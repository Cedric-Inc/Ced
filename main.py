class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """

        def do_cal(x, y, cal):
            if cal == '+':
                return x + y
            elif cal == '-':
                return x - y
            elif cal == '*':
                return x * y
            else:
                return int(x / y)

        cal = {'+', '-', '*', '/'}
        stack = []
        for i in tokens:
            if i not in cal:
                stack.append(int(i))
            else:
                x = stack.pop()
                y = stack.pop()
                stack.append(do_cal(int(y), int(x), i))
        return stack.pop()


s = Solution()
a = s.evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"])
print(a)