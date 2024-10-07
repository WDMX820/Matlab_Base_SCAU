# **数学建模实验报告**

## **Mathematical Modeling Experiment Report**

### 实验一  利用matlab对矩阵进行相关操作

**1、实验名称：利用matlab对矩阵进行相关操作**

**2、实验目的：掌握如何利用matlab计算矩阵的行列式、秩、逆矩阵，以及for循环语句。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190143134.png" alt="image-20241007190143134" style="zoom:67%;" />

```matlab
A = [1 2 3;4 5 7;7 4 1];
disp(['矩阵的行列式是',num2str(det(A))]);
disp(['矩阵的秩是',num2str(rank(A))]);
disp('矩阵的逆矩阵是');
disp(A^(-1));
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190223315.png" alt="image-20241007190223315" style="zoom:67%;" />

```matlab
x = [-3];
for i=2:20
    x(i) = (x(i-1)+3/x(i-1))/2;
end
disp('数列xn的前20项为');
disp(x);
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190320036.png" alt="image-20241007190320036" style="zoom:67%;" />

```matlab
m=5;
n=1;
x=zeros(5,5);
for i=1:5
    for j=1:5
        if i == j
            x(i,j) = 3;
        elseif  abs(i-j) == 1
            x(i,j) = 2;
        end
    end
end
x
```



## 实验二  数据拟合

**1、实验名称：数据的拟合**

**2、实验目的：**

**（1）直观了解拟合基本内容；**

**（2）掌握用数学软件求解拟合问题。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190545708.png" alt="image-20241007190545708" style="zoom:67%;" />

```matlab
t = [20.5 32.7 51.0 73.0 95.7];
R = [765 826 873 942 1032];
P = polyfit(t,R,1);
ti = 0:2:100;
Ri = polyval(P,ti);
plot(ti,Ri,t,R,"r*");
R60 = polyval(P,60)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190618410.png" alt="image-20241007190618410" style="zoom:67%;" />

```matlab
d = 300;
t = [0.25 0.5 1 1.5 2 3 4 6 8];
c = [19.21 18.15 15.36 14.10 12.89 9.32 7.45 5.24 3.01];
y = log(c);
a = polyfit(t,y,1)
k = -a(1)
v = d/exp(a(2))
c9 = polyval(a,9)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190641771.png" alt="image-20241007190641771" style="zoom:67%;" />

```matlab
x = [1 2 4 7 9 12 13 15 17];
f = [1.5 3.9 6.6 11.7 15.6 18.8 19.6 20.6 21.1];
sum1 = 0; sum2 = 0; sum3 = 0; sum4 = 0;
xi = 1:1:20;
a = polyfit(x,f,1);
fi = polyval(a,xi);
fx = polyval(a,x);
for k = 1:9
    sum1 = abs(fx(k)-f(k))*abs(fx(k)-f(k))+sum1;
end
subplot(2,2,1)
plot(x,f,'r*',xi,fi,'b-')
title('一次拟合');
a = polyfit(x,f,2);
fi = polyval(a,xi);
fx = polyval(a,x);
for k = 1:9
    sum2 = abs(fx(k)-f(k))*abs(fx(k)-f(k))+sum2;
end
subplot(2,2,2)
plot(x,f,'r*',xi,fi,'k-')
title('二次拟合');
a = polyfit(x,f,3);
fi = polyval(a,xi);
fx = polyval(a,x);
for k = 1:9
    sum3 = abs(fx(k)-f(k))*abs(fx(k)-f(k))+sum3;
end
subplot(2,2,3)
plot(x,f,'r*',xi,fi,'g-')
title('三次拟合');
a = polyfit(x,f,4);
fi = polyval(a,xi);
fx = polyval(a,x);
for k = 1:9
    sum4 = abs(fx(k)-f(k))*abs(fx(k)-f(k))+sum4;
end
subplot(2,2,4)
plot(x,f,'r*',xi,fi,'p-')
title('四次拟合');
sum1
sum2
sum3
sum4
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190735373.png" alt="image-20241007190735373" style="zoom:67%;" />

```matlab
x = [0.1 0.2 0.4 0.5 0.6 0.7 0.8 0.9 1];
y = [1.978 3.28 6.16 7.34 7.66 9.58 9.48 9.30 11.2];
sum1 = 0; sum2 = 0;
xi = 0.1:.1:1;
a = polyfit(x,y,2);
fi = polyval(a,xi);
fx = polyval(a,x);
for k = 1:9
    sum1 = abs(fx(k)-y(k))*abs(fx(k)-y(k))+sum1;
end
subplot(1,2,1);
plot(x,y,'r*',xi,fi,'k-')
title('二次拟合');
a = polyfit(x,y,3);
fi = polyval(a,xi);
fx = polyval(a,x);
for k = 1:9
    sum2 = abs(fx(k)-y(k))*abs(fx(k)-y(k))+sum2;
end
subplot(1,2,2);
plot(x,y,'r*',xi,fi,'b-')
title('二次拟合');
sum1
sum2
```



### 实验三  matlab曲线拟合

**1、实验名称：matlab曲线拟合**

**2、实验目的：利用matlab通过线性最小二乘法、多项式拟合对路障、电阻和录像机计数器等数据进行处理。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190812508.png" alt="image-20241007190812508" style="zoom:67%;" />

```matlab
%加速测试数据
clc;clear
y0=[0 2.7 5.5 8.3 11.1];
x0=[0 1.6 3.0 4.2 5.0];
a1=polyfit(x0,y0,1)
y1=polyval(a1,y0);
subplot(1,2,1)
plot(x0,y0,'o',x0,y1,'-')
title('加速测试数据');
%减速测试数据
y2=[11.1,8.3,5.5,2.7,0];
x1=[0,2.2,4.0,5.5,6.8];
a2=polyfit(x1,y2,1)
y3=polyval(a2,x1);
subplot(1,2,2)
plot(x1,y2,'o',x1,y3,'-')
title('减速测试数据');
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190835768.png" alt="image-20241007190835768" style="zoom:67%;" />

```matlab
t=[0,20,40,60,80,100,120,140,160,184];
n=[0000,1141,2019,2760,3413,4004,4545,5051,5525,6061];
f=inline('a(1)*n.^2+a(2)*n','a','n');
a=lsqcurvefit(f,[0 0],n,t)
%绘制曲线
y1=f(a,n);
plot(n,y1,n,t,'o');
legend('拟合曲线','原数据点','location','NorthWest');
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190853294.png" alt="image-20241007190853294" style="zoom:67%;" />

```matlab
clc;clear
t = [0.5 1 2 3 4 5 7 9];
vdata = [6.36 6.48 7.26 8.22 8.66 8.99 9.43 9.63];
V = 10;
fun = @(x,t)V-(V-x(1))*exp(-t/x(2));
x0 = [0,1];
x = lsqcurvefit(fun,x0,t,vdata);
disp(['V0的值是',num2str(x(1))]);
disp(['τ的值是',num2str(x(2))]);
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007190932832.png" alt="image-20241007190932832" style="zoom:67%;" />

```matlab
xi=0:1:10;
yi=xi.^3-xi.^2+5*xi-3;
C=rand(1,11);
y=yi+C;
subplot(2,2,1);
plot(xi,yi,'b-',xi,yi,'r*')
title('原函数');
a = [1,-1,5,-3];
a3 = polyfit(xi,y,3)
fi = polyval(a3,xi);
fx = polyval(a3,xi);
subplot(2,2,2);
plot(xi,y,'r*',xi,fi,'b-')
title('三次拟合');
a2 = polyfit(xi,y,2)
fi = polyval(a2,xi);
fx = polyval(a2,xi);
subplot(2,2,3);
plot(xi,y,'r*',xi,fi,'b-')
title('二次拟合');
a4 = polyfit(xi,y,4)
fi = polyval(a4,xi);
fx = polyval(a4,xi);
subplot(2,2,4);
plot(xi,y,'r*',xi,fi,'b-')
title('四次拟合');
```



### 实验四  matlab插值

**1、实验名称：matlab插值**

**2、实验目的：**

**（1）了解插值的基本内容;**

**（2）掌握通过插值的方法对给定函数进行插值计算。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191036098.png" alt="image-20241007191036098" style="zoom:67%;" />

```matlab
clc;clear
x=linspace(-6,6,100);
y=1./(1+x.^2);
x1=linspace(-6,6,5);
y1=1./(1+x1.^2);
subplot(2,2,1)
plot(x,y,x1,y1,'o')
title('选5个点做插值');
gtext('n=5');
x2=linspace(-6,6,11);
y2=1./(1+x2.^2);
subplot(2,2,2)
plot(x,y,x2,y2,'o')
title('选11个点做插值');
gtext('n=11')
x2=linspace(-6,6,21);
y2=1./(1+x2.^2);
subplot(2,2,3)
plot(x,y,x2,y2,'o')
title('选21个点做插值');
gtext('n=21')
x2=linspace(-6,6,41);
y2=1./(1+x2.^2);
subplot(2,2,4)
plot(x,y,x2,y2,'o');
title('选41个点做插值');
gtext('n=41')
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191108217.png" alt="image-20241007191108217" style="zoom:67%;" />

```matlab
clc;clear
l=[1604,1635,1640,1687,1655,1786,1723,1523,1465,1200];
d=[957,963,969,974,975,977,986,993,998,1000];
up=l-d;
year=2011:2020;
p1 = pchip(year, up, 2021:2023)  %分段三次Hermit插值预测
p2 = spline(year, up, 2021:2023) %三次样条插值预测
figure(4);
plot(year, up,'o',2021:2023,p1,'r*-',2021:2023,p2,'bx-')
legend('样本点','三次Hermit插值预测','三次样条插值预测','Location','southwest')
```



### 实验五­  山区地貌和排沙量估算

**1、实验名称：山区地貌和排沙量估算**

**2、实验目的：**

**（1）掌握最近邻点插值等方法并对山区地貌绘制地貌图和等高线图;**

**（2）掌握二维插值方法并利用其对各时刻排沙量情况进行估计。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191216162.png" alt="image-20241007191216162" style="zoom:67%;" />

```matlab
clc,clear
x=1200:400:4000;
y=1200:400:3600;
subplot(2,2,1);
[xx,yy]=meshgrid(x,y);
z=[1480 1500 1550 1510 1430 1300 1200 980;
    1500 1550 1600 1550 1600 1600 1600 1550;
    1500 1200 1100 1550 1600 1550 1380 1070;
    1500 1200 1100 1350 1450 1200 1150 1010;
    1390 1500 1500 1400 900 1100 1060 950;
    1320 1450 1420 1400 1300 700 900 850;
    1130 1250 1280 1230 1040 900 500 700];
surf(xx,yy,z);
hold on;
[c,h]=contour(xx,yy,z,6);
title('原始图');
%clabel(c,h)   %值标签
%最近邻点插值
xi=1200:4000;
yi=1200:3600;
subplot(2,2,2);
[xi,yi]=meshgrid(xi,yi);
zi=interp2(x,y,z,xi,yi,'nearest');
% surf(xi,yi,zi);
% hold on;
% [c,h]=contour(xi,yi,zi,6);
meshc(xi,yi,zi)
title('最近邻点插值');
%双线性插值
xi=1200:4000;
yi=1200:3600;
subplot(2,2,3);
[xi,yi]=meshgrid(xi,yi);
zi=interp2(x,y,z,xi,yi,'linear');
meshc(xi,yi,zi)
title('双线性插值');
%双三次插值
xi=1200:4000;
yi=1200:3600;
subplot(2,2,4);
[xi,yi]=meshgrid(xi,yi);
zi=interp2(x,y,z,xi,yi,'cubic');
meshc(xi,yi,zi)
title('双三次插值');
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191322273.png" alt="image-20241007191322273" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191331130.png" alt="image-20241007191331130" style="zoom:67%;" />

```matlab
% 问题一
% v:水流量S:含沙量；V:排沙量?
% 假设水流量和含沙量都是连续的，某一时刻的排沙量V=v(t)S(t)
% 先已知某些时刻的水流量和含沙量，给出估计任意时刻的排沙量及总排沙量
% 总排沙量是对排沙量做积分
% 时间8:00-20:00
format compact;
clc,clear;
load data3.txt
liu = data3([1,3],:); liu=liu';liu=liu(:); % 提出水流量并按照顺序变成列向量
sha = data3([2,4],:); sha=sha';sha=sha(:); % 提出含沙量并按照顺序变成列向量
y=sha.*liu;y=y'; % 计算排沙量，变成行向量
i=1:24;
t=(12*i-4)*3600;
t1=t(1);t2=t(end);
pp=csape(t,y); % 进行三次样条插值
xsh=pp.coefs; % 求得插值多项式的系数矩阵,每一行是一个区间上多项式的系数
TL = quadl(@(tt)fnval(pp,tt),t1,t2) % 求总排沙量的积分运算
t0=t1:0.1:t2;
y0=fnval(pp,t0);
plot(t,y,'+',t0,y0)
% 问题二：确定排沙量和水流量的关系
% 画出排沙量和水流量的散点图

clc,clear;
load data3.txt
liu = data3([1,3],:); liu=liu';liu=liu(:); % 提出水流量并按照顺序变成列向量
sha = data3([2,4],:); sha=sha';sha=sha(:); % 提出含沙量并按照顺序变成列向量
y=sha.*liu; % 计算排沙量，这里是列向量
subplot(1,2,1),plot(liu(1:11),y(1:11),'*')
subplot(1,2,2),plot(liu(12:24),y(12:24),'*')
%以下是第一阶段的拟合
for j=1:2
    nihe1{j}=polyfit(liu(1:11),y(1:11),j); % 拟合多项式,系数排列从高次幂到低次
    yhat1{j}=polyval(nihe1{j},liu(1:11)); % 求预测值
    cha1(j)=sum((y(1:11)-yhat1{j}).^2); % 求误差平方和
    rmse1(j)=sqrt(cha1(j)/(10-j)); % 求剩余标准差
end
celldisp(nihe1) % 显示细胞数组的所有元素
rmse1
%以下是第二阶段的拟合
for j=1:2
    nihe2{j}=polyfit(liu(12:24),y(12:24),(j)); % 使用细胞数组
    yhat2{j}=polyval(nihe2{j},liu(12:24)); % 求预测值
    cha2(j)=sum((y(12:24)-yhat2{j}).^2); % 求误差平方和
    rmse2(j)=sqrt(cha2(j)/(11-j)); % 求剩余标准差
end
celldisp(nihe2) % 显示细胞数组的所有元素
rmse2
format % 恢复默认短小数的显示格式
```



### 实验六­  最小二乘法和拟合的实际应用

**1、实验名称：最小二乘法和拟合的实际应用**

**2、实验目的：通过汽车刹车距离、划艇比赛和席位分配熟悉拟合和Q值法。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191419102.png" alt="image-20241007191419102" style="zoom:67%;" />

```matlab
%模型：d=t1v+kv2 其中，d为刹车距离，变量v为车速
%参数 t1为反应时间，参数k为比例系数。取经验值t1=0.75秒
%拟合
clear;clc;
v=[29.3 44.0 58.7 73.3 88.0 102.7 117.3]; %英尺/秒
d=[44 78 124 186 268 372 506]; %最大实际刹车距离（英尺）
%d=[42 73.5 116 173 248 343 464]; %最大实际刹车距离（英尺）
y=(d-0.75*v)./v.^2;
k=polyfit(v,y,0)
dd=0.75*v+k*v.^2; %计算刹车距离
t=d./v; %计算刹车时间
format short g;
[v',d',round(10*[dd',t'])/10]
vh=[20 30 40 50 60 70 80]; %英里/小时
plot(vh,d,'r+',vh,dd,'b-');
title('实际和计算刹车距离的比较');
axis([20,80,0,510]);
xlabel('v 英里/小时');
ylabel('d 英尺');
%最小二乘法
clear;clc;
v=[29.3 44.0 58.7 73.3 88.0 102.7 117.3]; %英尺/秒
d=[44 78 124 186 268 372 506]; %最大实际刹车距离（英尺）
%d=[42 73.5 116 173 248 343 464]; %最大实际刹车距离（英尺）
fun=@(k,v)0.75*v+k*v.^2;
k=lsqcurvefit(fun,29,v,d)
dd=0.75*v+k*v.^2; %计算刹车距离
t=d./v; %计算刹车时间
format short g;
[v',d',round(10*[dd',t'])/10]
vh=[20 30 40 50 60 70 80]; %英里/小时
plot(vh,d,'r+',vh,dd,'b-');
title('实际和计算刹车距离的比较');
axis([20,80,0,510]);
xlabel('v 英里/小时');
ylabel('d 英尺');
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191449970.png" alt="image-20241007191449970" style="zoom:67%;" />

```matlab
%拟合
clear;clc;
n=[1 2 4 8];
t=[7.21 6.88 6.32 5.84];
logt=log(t);
logn=log(n);
p=polyfit(logn,logt,1);
b=p(1)
a=exp(p(2))
x=0:0.01:10;
y=a*x.^b;
plot(n,t,'r+',x,y,'b-');
title('实际和计算划艇成绩的比较');
axis([0,10,5,11]);
xlabel('n 桨手人数');
ylabel('t 平均成绩');
[n',t',(a*n.^b)']
%最小二乘法
clear;clc;
n=[1 2 4 8];
t=[7.21 6.88 6.32 5.84];
logt=log(t);
logn=log(n);
fun=@(x,n)x(1)+x(2)*logn;
x0=[0,1];
x=lsqcurvefit(fun,x0,n,t)
b=x(1);
a=exp(x(2));
x1=0:0.01:10;
y=a*x1.^b;
plot(n,t,'r+',x1,y,'b-');
title('实际和计算划艇成绩的比较');
axis([0,10,5,11]);
xlabel('n 桨手人数');
ylabel('t 平均成绩');
[n',t',(a*n.^b)']
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191520026.png" alt="image-20241007191520026" style="zoom:67%;" />

```matlab
clc,clear
P=200;
N=21;     %席位总数
n=[103 63 34];  %原来的
p=fix(n./P.*N);   %向0方向取整
m=N-sum(p);
i=1;
while i<=m
    Q=n.^2./(p.*(p+1))
    qq=find(Q==max(Q));    %最大Q则分配
    p(qq)=p(qq)+1
    m=m-1;
end 
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191611076.png" alt="image-20241007191611076" style="zoom:67%;" />

```matlab
clc,clear
year = [1990 1991 1992 1993 1994 1995 1996];
profit = [70 122 144 152 174 196 202];
plot(year,profit,'*');   %原始图
a = polyfit(year,profit,1);    %一次拟合
yearr = [1997 1998];   %预测年数
answer = polyval(a,yearr,1);      
hold on;
plot([year yearr],[profit answer],'*');
%计算误差
[~,n] = size(profit);
avg_profit = sum(profit)/n;
a1 = sum((polyval(a,year,1)-profit).^2)
a = polyfit(year,profit,2);
a2 = sum((polyval(a,year,2)-profit).^2)
a = polyfit(year,profit,3);
a3 = sum((polyval(a,year,3)-profit).^2)
if a1<=a2 && a1<=a3
    disp('应采用一次多项式');
elseif a1==a2==a3
    disp('一次、二次和三次多项式都可以');
elseif a2<=a1 && a2<=a3
    disp('应采用二次多项式');
elseif a3<=a1 && a3<=a1
    disp('应采用三次多项式');
end
disp(['1997,1998的预测值为',num2str(answer)]);
```



### 实验七  银行证券与奶制品加工

**1、实验名称：银行证券与奶制品加工**

**2、实验目的：通过建立规划模型求解如何安排生产计划使得最后获利最大。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191638431.png" alt="image-20241007191638431" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191701016.png" alt="image-20241007191701016" style="zoom:67%;" />

```matlab
%第一小问
clear,clc 
f=-1*[0.043,0.027,0.025,0.022,0.045];
intcon=(1:5);
A=[0,-1,-1,-1,0;
    1,1,1,1,1;
    0.6,0.6,-0.4,-0.4,3.6;
    4,10,-1,-2,-3];
b=[-400;1000;0;0];
Aeq=[]; beq=[];
lb=zeros(5,1);
ub=[Inf;Inf];
[x,fval]= linprog(f,A,b,Aeq,beq,lb,ub);
x
-1*fval
%第二小问
clear,clc 
f=-1*[0.043,0.027,0.025,0.022,0.045,-0.0275];
intcon=(1:6);
A=[0,-1,-1,-1,0,0;
    1,1,1,1,1,-1;
    0.6,0.6,-0.4,-0.4,3.6,0;
    4,10,-1,-2,-3,0;
    0,0,0,0,0,1];
b=[-400;1000;0;0;100];
Aeq=[]; beq=[];
lb=zeros(5,1);
ub=[Inf;Inf];
[x,fval]= linprog(f,A,b,Aeq,beq,lb,ub);
x
-1*fval
%第三小问
%A变成4.5%
clear,clc 
f=-1*[0.045,0.027,0.025,0.022,0.045];
intcon=(1:5);
A=[0,-1,-1,-1,0;
    1,1,1,1,1;
    0.6,0.6,-0.4,-0.4,3.6;
    4,10,-1,-2,-3];
b=[-400;1000;0;0];
Aeq=[]; beq=[];
lb=zeros(5,1);
ub=[Inf;Inf];
[x,fval]= linprog(f,A,b,Aeq,beq,lb,ub);
x
-1*fval
%A变成4.8%
clear,clc 
f=-1*[0.043,0.027,0.023,0.022,0.045];
intcon=(1:5);
A=[0,-1,-1,-1,0;
    1,1,1,1,1;
    0.6,0.6,-0.4,-0.4,3.6;
    4,10,-1,-2,-3];
b=[-400;1000;0;0];
Aeq=[]; beq=[];
lb=zeros(5,1);
ub=[Inf;Inf];
[x,fval]= linprog(f,A,b,Aeq,beq,lb,ub);
x
-1*fval
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191739097.png" alt="image-20241007191739097" style="zoom:67%;" />

```matlab
%获利最大
clc,clear
fun = @(x)-(72*x(1)+64*x(2));
x0 = zeros(1,2);
A = [1 1;12 8;3 0];
b = [50 480 100]';
Aeq = []; beq = [];
lb = zeros(1,2);
ub = [];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
disp('生产A1桶');
disp(x(1))
disp('生产A2桶');
disp(x(2))
disp('利润为');
disp(-fval)
%可以买牛奶
clear,clc
fun = @(x)-(72*(x(1)+x(3))+64*(x(2)+x(4))-35*(x(3)+x(4)));
x0 = zeros(1,4);
A = [1 1 0 0;12 8 12 8;3 0 3 0];
b = [50 480 100]';
Aeq = []; beq = [];
lb = zeros(1,4);
ub = [];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
disp('生产A1桶');
disp(x(1))
disp('生产A2桶');
disp(x(2))
disp('新买牛奶');
disp(x(3)+x(4))
disp('利润为');
disp(-fval)
%可以聘用工人
clc,clear
sum = [];
for k=0:0.1:10
fun = @(x)-(72*x(1)+64*x(2)-k*x(3));
x0 = zeros(1,3);
A = [1 1 0;12 8 -1;3 0 0];
b = [50 480 100]';
Aeq = []; beq = [];
lb = zeros(1,3);
ub = [];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
sum = [sum,-fval];
end
x = 0:0.1:10;
plot(x,sum);
%时间增加1单位，利润增长2元,所以聘用临时工人付出的工资最多每小时2元.
%A1涨价
clear,clc
fun = @(x)-(90*x(1)+64*x(2));
x0 = zeros(1,2);
A = [1 1;12 8;3 0];
b = [50 480 100]';
Aeq = []; beq = [];
lb = zeros(1,2);
ub = [];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
disp('生产A1桶');
disp(x(1))
disp('生产A2桶');
disp(x(2))
disp('利润为');
disp(-fval)
```



### 实验八  生产计划与投资组合

**1、实验名称：生产计划与投资组合**

**2、实验目的：通过建立规划模型，求解怎样安排生产才能使得获利最大。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191901752.png" alt="image-20241007191901752" style="zoom:67%;" />

```matlab
fun = @(x)-(10*x(1)+9*x(2));
x0 = zeros(1,2);
A = [6 5;10 20];
b = [60 150]';
Aeq = []; beq = [];
lb = zeros(1,2);
ub = [8 inf];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
disp(['生产甲饮料',num2str(x(1)),'百箱']);
disp(['生产乙饮料',num2str(x(2)),'百箱']);
disp(['利润为',num2str(-fval),'万元']);
%%可购买原料
fun = @(x)-(10*x(1)+9*x(2)-0.8*x(3));
x0 = zeros(1,3);
A = [6 5 -1;10 20 0];
b = [60 150]';
Aeq = []; beq = [];
lb = zeros(1,2); ub = [8 inf];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
disp(['生产甲饮料',num2str(x(1)),'百箱']);
disp(['生产乙饮料',num2str(x(2)),'百箱']);
disp(['新购买原料',num2str(x(3)),'千克']);
disp(['利润为',num2str(-fval),'万元']);
%%甲饮料获利增加
fun = @(x)-(11*x(1)+9*x(2));
x0 = zeros(1,2);
A = [6 5;10 20];
b = [60 150]';
Aeq = []; beq = [];
lb = zeros(1,2); ub = [8 inf];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
disp(['生产甲饮料',num2str(x(1)),'百箱']);
disp(['生产乙饮料',num2str(x(2)),'百箱']);
disp(['利润为',num2str(-fval),'万元']);
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007191953181.png" alt="image-20241007191953181" style="zoom:67%;" />

```matlab
clc;clear;
k = 0.04;
hold on
while k < 0.25
    c = [0,0,0,0,0,1];
    A = [0,0.025,0,0,0,-1
        0,0,0.015,0,0,-1
        0,0,0,0.055,0,-1
        0,0,0,0,0.026,-1
        -0.05,-0.27,-0.19,-0.185,-0.185,0];
    b = [0,0,0,0,-k];
    Aeq = [1,1.01,1.02,1.045,1.065,0];          
    beq = 1;
    LB=zeros(6,1); 
    [x,Q] = linprog(c,A,b,Aeq,beq,LB);
    Q=-Q;                            
    plot(k,Q,'b*');
    k = k+0.01;
end 
xlabel('h'),ylabel('Q')
 c = [0,0,0,0,0,1];
    A = [0,0.025,0,0,0,-1; 0,0,0.015,0,0,-1; 0,0,0,0.055,0,-1
        0,0,0,0,0.026,-1; -0.05,-0.27,-0.19,-0.185,-0.185,0];
b = [0,0,0,0,-k];
    Aeq = [1,1.01,1.02,1.045,1.065,0];          
    beq = 1;
    LB=zeros(6,1); 
    [x,Q] = linprog(c,A,b,Aeq,beq,LB);
    Q=-Q;                            
    plot(k,Q,'b*');
    k = k+0.01;
end 
xlabel('h'),ylabel('Q')
 c = [0,0,0,0,0,1];
    A = [0,0.025,0,0,0,-1
        0,0,0.015,0,0,-1
        0,0,0,0.055,0,-1
        0,0,0,0,0.026,-1
        -0.05,-0.27,-0.19,-0.185,-0.185,0];
    b = [0,0,0,0,-0.2];
    Aeq = [1,1.01,1.02,1.045,1.065,0];          
    beq = 1;
    LB=zeros(6,1); 
    [x,Q] = linprog(c,A,b,Aeq,beq,LB)
```



### 实验九  非线性规划

**1、 实验名称：非线性规划**

**2、 实验目的：根据题目所给目标函数和约束条件进行非线性规划计算。**

**3、 实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192030595.png" alt="image-20241007192030595" style="zoom:67%;" />

```matlab
clc;clear
fun = @(x)-2*x(1)-x(2);
x0 = zeros(1,2);
A = []; 
b = [];
Aeq = []; 
beq = [];
lb = zeros(1,2);
ub = [5 10];
[x,fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,'fun2');
disp(['x1=',num2str(x(1))]);
disp(['x2=',num2str(x(2))]);
disp(['最小值为',num2str(fval)]);
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192049354.png" alt="image-20241007192049354" style="zoom:67%;" />

```matlab
clc;clear
fun=@(x)x(1).^2+2*x(2).^2-2*x(1)*x(2)-4*x(1)-12*x(2);
x0=[0,0];
A=[1,1;-1,2;2,1];
b=[2,2,3];
Aeq=[]; beq=[];
[x,y]=fmincon(fun,x0,A,b,Aeq,beq);
disp(['x1=',num2str(x(1))]);
disp(['x2=',num2str(x(2))]);
disp(['最小值为',num2str(y)]);
```



### 实验十  液体混合加工

**1、实验名称：液体混合加工**

**2、实验目的：通过题目所给数据和限制要求建立数学模型并规划求解。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192126331.png" alt="image-20241007192126331" style="zoom:67%;" />

```matlab
% 目标函数
% max = y1*(9-6*x1-16*x2-15*x4)+y2*(15-6*x1-16*x2-15*x4)+(9-10)*z1+(15-10)*z2
% 约束条件
% x1 + x2 +x4 == 1  % 原料所占的比例之和
% x4(y1 + y2) <= 100   z1 + z2 <= 250   % 原料供应限制
% y1+z1 <= 300   y2+z2 <= 500 % 市场需求的限制
% （(3*x1 + x2 + x4）* y1)/(y1+z1) <= 2.5
% （（3*x1 + x2 + x4）* y2）/(y2+z2) <= 1.5;  % 产品含硫量比例限制
% 转化为矩阵x = [x1, x2, x4, y1, y2, z1, z2]
% 符号说明 y1 y2 为A B产品来自混合池的吨数
% z1 z2 为A B产品来自原料丙的吨数
% x1 x2 x4 分别是原料 甲 已 丁 所占的比例数
clear; clc;
x0 = [0.1; 0; 0; 0; 0; 0; 0];
A = [0 0 0 0 0 1 1; 0 0 0 1 0 1 0; 0 0 0 0 1 0 1;];
b = [250; 300; 500];
% 等式现行约束
Aeq = [1 1 1 0 0 0 0];
beq = 1;
vlb = [0 0 0 0 0 0 0];
vub = [1 1 1 inf inf inf inf];
[x, fval] = fmincon('fun5', x0, A, b, Aeq, beq, vlb, vub, 'con5');
disp(['利润为',num2str(-fval),'万元']);
```



### 实验十一  合金强度与含碳量

**1、 实验名称：合金强度与含碳量**

**2、 实验目的：建立回归方程对合金强度与其中的碳含量的关系进行分析。**

**3、 实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192211965.png" alt="image-20241007192211965" style="zoom:67%;" />

```matlab
x = [0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18];  %输入x值
y = [42 41.5 45 45.5 45 47.5 49 55 50]';  %输入y值
X = [ones(9,1) x'];  %自变量组成的矩阵
subplot(3,2,1:2)   %画在一张图
plot(x,y)
title('原数据散点图')
[b,bint,r,rint,status] = regress(y,X,0.05)%以显著水平0.05进行回归方程构建
% b：回归系数
% bint：回归系数的区间估计，区间越小精度越高
% r：残差
% rint：置信区间
% stats：4个检验回归模型的统计量R方，F统计量，检验p值，误差方差的估计
subplot(3,2,3)
rcoplot(r,rint);  %做残差分析图
title('未去除离群点时残差分析图')
y_result = b(1)*x+b(2);
subplot(3,2,4)
plot(x,y_result);
title('未去除离群点时回归方程图')
contain0 = (rint(:,1)<0 & rint(:,2)>0);   %contain0为0即为异常
idx = find(contain0==false);    
y(idx)=(y(idx-1)+y(idx+1))/2;  %均值替代（前后）
% y(8)=(y(1)+y(2)+y(3)+y(4)+y(5)+y(6)+y(7)+y(8)+y(9))/9; %均值替代（全部均值）
[b2,bint2,r2,rint2,status2] = regress(y,X);  %使用regress回归
subplot(3,2,5)
rcoplot(r2,rint2);%做残差分析图
title('去除离群点时残差分析图')
y_result = b(1)*x+b(2);
subplot(3,2,4)
plot(x,y_result);
title('未去除离群点时回归方程图')
contain0 = (rint(:,1)<0 & rint(:,2)>0);   %contain0为0即为异常
idx = find(contain0==false);    
y(idx)=(y(idx-1)+y(idx+1))/2;  %均值替代（前后）
% y(8)=(y(1)+y(2)+y(3)+y(4)+y(5)+y(6)+y(7)+y(8)+y(9))/9;  %均值替代（全部均值）
[b2,bint2,r2,rint2,status2] = regress(y,X);  %使用regress回归
subplot(3,2,5)
rcoplot(r2,rint2);%做残差分析图
title('去除离群点时残差分析图')
y_result2 = b2(1)*x+b2(2);
subplot(3,2,6)
plot(x,y_result2);
title('去除离群点时回归方程图')
status
status2
```



### 实验十二  多项式回归预测

**1、实验名称：多项式回归预测**

**2、实验目的：利用二次多项式回归等方法计算和预测实际问题。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192307351.png" alt="image-20241007192307351" style="zoom:67%;" />

```matlab
%第一题：零件曲线——二次多项式回归
%polyfit方法
clc,clear
x = 0:2:20;  %数列方法
y = [0.6 2 4.4 7.5 11.8 17.1 23.3 31.2 39.6 49.7 61.7];
[p,S] = polyfit(x,y,2);   %二次多项式拟合
x1 = 0:0.05:20;
y1 = polyval(p,x1);   %求解对应y值
plot(x,y,'*',x1,y1,'r-')   %画图
legend('散点图','回归方程图')
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192352452.png" alt="image-20241007192352452" style="zoom:67%;" />

```matlab
clear,clc
x1 = [120 140 190 130 155 175 125 145 180 150];
x2 = [100 110 90 150 210 150 250 270 300 250];
y = [102 100 120 77 46 93 26 69 65 85];
rstool([x1' x2'],y','quadratic')
%regress
X = [ones(10,1) x1' x2' (x1.^2)' (x2.^2)'];
[b,bint,r,rint,stats] = regress(y',X);
t1=170;t2=160;
y = b(1)+b(2)*t1+b(3)*t2+b(4)*t1.^2+b(5)*t2.^2
```



### 实验十三   回归分析

**1、实验名称：回归分析**

**2、实验目的：利用matlab建立回归分析模型并对商品销售量及其影响因素进行分析。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192439653.png" alt="image-20241007192439653" style="zoom:67%;" />

```matlab
clear;clc;
load 'test13.txt';
x1 = test13(:,1);
x2 = test13(:,2);
x3 = test13(:,3);
x4 = test13(:,4);
y = test13(:,5);
X = [x1 x2 x3 x4];
disp(['回归方程: Y=',num2str(a(1)),'+',num2str(a(2)),'*x1','+',num2str(a(3)),'*x2','+',num2str(a(4)),'*x4'])
```



### 实验十四   Q型聚类分析

**1、实验名称：Q型聚类分析**

**2、实验目的：利用matlab建立回归分析模型并对国家和地区进行聚类分析。**

**3、实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192525699.png" alt="image-20241007192525699" style="zoom:67%;" />

```matlab
clear,clc
load 'X.txt';
[m,n] = size(X);
X1 = zscore(X);
y = pdist(X1,'euclidean');  %欧氏距离
z = linkage(y);  %画聚类树
cophenet(z,y) %共性相关，值越大表明树对距离拟合越好
% subplot(2,1,1);
[h,t] = dendrogram(z);
T = cluster(z,'maxclust',5);%创建阈值
for i=1:5
    tm = find(T==i);
    tm = reshape(tm,1,length(tm));
    fprintf('第%d类的有%s\n',i,int2str(tm));
end
title('欧几里德距离画聚类树');
```



### 实验十五   R型聚类分析

**1、 实验名称：R型聚类分析**

**2、 实验目的：通过R型聚类分析对题目所给实际数据和目标对象进行聚类分类。**

**3、 实验要求与过程：**

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007192616549.png" alt="image-20241007192616549" style="zoom:67%;" />

```matlab
%指标聚类降维
clc,clear
a=load('test15.txt'); %把原始的x1,x2,x3,x4,y的数据保存在纯文本文件a.txt中
b=zscore(a);
r=corrcoef(b);
d=pdist(b','correlation');
z=linkage(d,'average');
subplot(2,1,1);
h=dendrogram(z);
set(h,'Color','k','LineWidth',1.3);
title('指标聚类降维图');
T=cluster(z,'maxclust',3)
for i=1:3
    tm=find(T==i);
    tm=reshape(tm,1,length(tm));
    fprintf("第%d类的有%s\n",i,int2str(tm));
end
fprintf("\n");
%样本聚类
clear
a=load('test16.txt'); %把原始的x1,x2,x3,x4,y的数据保存在纯文本文件a.txt中
a=zscore(a);  %数据标准化
y=pdist(a);  %求对象间的欧氏距离，每行是一个对象
z=linkage(y,'average');  %按类平均法聚类
subplot(2,1,2);
h=dendrogram(z);   %画聚类图
set(h,'Color','k','LineWidth',1.3);  %黑色，1.3粗细
title('样本聚类示意图');
for k=3:6
    fprintf("划分成%d类的结果如下：\n",k)
    T=cluster(z,'maxclust',k); %划分成k类
    for i=1:k
        tm=find(T==i);  %求第i类对象
        tm=reshape(tm,1,length(tm));
        fprintf("第%d类的有%s\n",i,int2str(tm));
    end
    fprintf("\n")
end
```







