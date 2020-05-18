# -*- coding: utf-8 -*-
def grad_methon(old_w,old_grad):
    w=old_w-0.01*old_grad;
    return w

if __name__=='__main__':
    print("main");
    x=grad_methon(5,2)
    print(x)