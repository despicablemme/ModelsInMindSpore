from mindspore.common.parameter import Parameter


class A:
    def __init__(self, val):
        self.val = val


a = A(1)
b = a

b.val = 2
print(a.val)