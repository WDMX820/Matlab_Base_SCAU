
%% 因子分析
clc,clear

ssgs = [0.6657 	0.0000 	0.0092 	0.0021 	0.0198 	0.0081 	0.0026 	0.0084 	0.0000 	0.0000 	0.0000 	0.0000 	0.0000 	0.0047 ;
0.6579 	0.0000 	0.0059 	0.0062 	0.0198 	0.0132 	0.0032 	0.0155 	0.0000 	0.0000 	0.0035 	0.0000 	0.0000 	0.0047 ;
0.6552 	0.0000 	0.0101 	0.0072 	0.0198 	0.0146 	0.0029 	0.0165 	0.0000 	0.0000 	0.0015 	0.0000 	0.0000 	0.0047 ;
0.6444 	0.0000 	0.0000 	0.0107 	0.0198 	0.0198 	0.0017 	0.0324 	0.0000 	0.0000 	0.0061 	0.0000 	0.0000 	0.0047 ;
0.6290 	0.0000 	0.0000 	0.0094 	0.0198 	0.0251 	0.0020 	0.0154 	0.0000 	0.0000 	0.0036 	0.0000 	0.0000 	0.0047 ;
0.6138 	0.0000 	0.0074 	0.0166 	0.0198 	0.0350 	0.0035 	0.0055 	0.0000 	0.0000 	0.0021 	0.0000 	0.0000 	0.0047 ;
0.8705 	0.0000 	0.0519 	0.0201 	0.0000 	0.0406 	0.0000 	0.0078 	0.0025 	0.0000 	0.0066 	0.0000 	0.0000 	0.0000 ;
0.7946 	0.0000 	0.0942 	0.0000 	0.0153 	0.0305 	0.0000 	0.0000 	0.0000 	0.0000 	0.0136 	0.0007 	0.0236 	0.0000 ;
0.7668 	0.0000 	0.0000 	0.0471 	0.0122 	0.0619 	0.0237 	0.0328 	0.0100 	0.0197 	0.0110 	0.0000 	0.0000 	0.0000 ;
0.6933 	0.0000 	0.0999 	0.0632 	0.0087 	0.0393 	0.0174 	0.0387 	0.0000 	0.0000 	0.0117 	0.0000 	0.0000 	0.0039 ;
0.6518 	0.0210 	0.1452 	0.0827 	0.0052 	0.0618 	0.0042 	0.0107 	0.0011 	0.0000 	0.0000 	0.0004 	0.0000 	0.0000 ;
0.5901 	0.0286 	0.1253 	0.0870 	0.0000 	0.0616 	0.0288 	0.0473 	0.0000 	0.0000 	0.0127 	0.0000 	0.0000 	0.0000 ;
0.6588 	0.0000 	0.0967 	0.0712 	0.0156 	0.0644 	0.0206 	0.0218 	0.0000 	0.0000 	0.0079 	0.0000 	0.0000 	0.0036 ;
0.6247 	0.0338 	0.1228 	0.0823 	0.0066 	0.0923 	0.0050 	0.0047 	0.0162 	0.0000 	0.0016 	0.0000 	0.0000 	0.0000 ;
0.6765 	0.0000 	0.0737 	0.0000 	0.0198 	0.1115 	0.0239 	0.0251 	0.0020 	0.0138 	0.0418 	0.0011 	0.0000 	0.0000 ;
0.6158 	0.0000 	0.1095 	0.0735 	0.0177 	0.0750 	0.0262 	0.0327 	0.0000 	0.0000 	0.0094 	0.0006 	0.0000 	0.0047 ;
0.6171 	0.0000 	0.1237 	0.0587 	0.0111 	0.0550 	0.0216 	0.0509 	0.0141 	0.0286 	0.0070 	0.0010 	0.0000 	0.0000 ;
0.5981 	0.0000 	0.0768 	0.0541 	0.0173 	0.1005 	0.0604 	0.0218 	0.0035 	0.0097 	0.0450 	0.0012 	0.0000 	0.0000 ];
n = size(ssgs, 1);
x = ssgs(:, [1:4]);
KMO(x)

y = ssgs(:, 5);
x = zscore(x);% 对前四个收益指标标准化处理
r = corrcoef(x); % 求相关系数矩阵 
[vec1, val, con1] = pcacov(r); % 进行主成分分析的相关计算
f1 = repmat(sign(sum(vec1)), size(vec1, 1), 1);
vec2 = vec1.*f1; % 特征向量正负号转换
f2 = repmat(sqrt(val)', size(vec2, 1), 1);
a = vec2 .* f2 % 求初等载荷矩阵
num = input('请选择主因子个数'); % 选择主因子个数
am = a(:, [1:num]); % 提出num个主因子的载荷矩阵
[bm, t] = rotatefactors(am, 'method', 'varimax'); %am旋转变换，bm为旋转后的载荷阵
bt = [bm, a(:, num+1:end)]; % 旋转后全部因子的载荷矩阵，前两个旋转，后面不旋转
con2 = sum(bt.^2); % 计算因子贡献
check = [con1, con2'/sum(con2) * 100]; % 该语句是领会旋转意义，con1是未旋转前的贡献率
rate = con2(1:num)/sum(con2); % 计算因子贡献率
coef = inv(r) * bm; % 计算得分函数的系数
score = x * coef; % 计算各个因子的得分
weight = rate/sum(rate); % 计算得分的权重
Tscore = score * weight' % 对各个因子的得分进行加权求和，即求各企业的综合得分
[STscore, ind] = sort(Tscore, 'descend') % 对企业进行排序
display = [score(ind, :)'; STscore'; ind']; % 显示排序结果
[ccoef, p] = corrcoef([Tscore, y]) % 计算F与资产负债的相关性系数
[d, dt, e, et, stats] = regress(Tscore, [ones(n, 1), y]); % 计算F与资产负债的方程
d, stats % 显示回归系数，和相关统计量的值




%% 二分求根

%二分求根算法实现：给一个f，一个左区间，一个右区间
%函数格式代码，包含注释
function G=EF(f,a,b)     %定义EF函数
lzc=10^-3;     %提前设置限制条件
x0=(a+b)/2;     %取a和b的中间值
if abs(b-a)<lzc      %判断ab的差值是否小于限制停止条件
    G=x0;      %将x0赋值给G
elseif f(a)*f(x0)<0         %判断a的函数值与x0的函数值是否小于0
    b=x0;      %将x0赋值给b
    EF(f,a,b)        
else
    a=x0;      %将x0赋值给a
    EF(f,a,b)
end    %结束




%% 高斯消元函数封装

function G=GSXY(a,b)
lzc=size(a,1);
G=zeros(lzc,1);
for ssh=1:lzc-1
    T=[];
    a=a(ssh:lzc,ssh);
    m=find(abs(a)==max(abs(a)));
    a([ssh,m(1)+ssh-1],:)=a([ssh+m(1)-1,ssh],:);
    [b(ssh),b(ssh+m(1)-1)]=deal(b(m(1)+ssh-1),b(ssh))
    for ssh=1:lzc-ssh
        T(ssh)=-a(ssh+ssh,ssh)/a(ssh,ssh);
        a(ssh+ssh,:)=a(ssh+ssh,:)+a(ssh,:)*T(ssh);
        b(ssh+ssh)=b(ssh+ssh)+b(ssh)*T(ssh);
    end
end
for ssh=lzc:-1:ssh+1
    sum=0;
    for ssh=lzc:-1:ssh+1
        sum=sum+G(ssh)*a(ssh,ssh);
    end
    G(ssh)=(b(ssh)-sum)/a(ssh,ssh)    
end




%% 主成分分析-KMO检验

function kmo=KMO(x)
    R=corrcoef(x); % 简单相关系数
    P=partialcorr(x);  %偏相关系数
    R_1=R-eye(size(R));  %简单相关系数减去对角线上的1
    P_1=P-eye(size(P));  %偏相关系数减去对角线上的1
    KMO=sum(R_1(:).^2)/(sum(R_1(:).^2)+sum(P_1(:).^2))
end




%% 拉格朗日插值算法

%拉格朗日插值算法的实现：给样本点和代插节点算插值的值
%函数格式代码，包含注释
function L=lglr(x,y,x0)   % 定义函数
a=x0;     % 将x0赋值给a
n=length(x);     % 将x的长度赋值给n
a0=zeros(1,n);     % 事先建立相应大小的全零矩阵
for a=1:n         % 使用for循环遍历
    xx=x(a);    % 获取x(a)并赋值给xx
    x1=x;      % 将x赋给x1
    x1(a)=[];   % 获取x中去除点x(a)后的数据
    shang=1;     % 先定义一个shang为1
    xia=1;     % 先定义一个xia为1
    for b=1:n-1      % 使用for循环遍历
        shang=shang*(a-x1(b));   % 连乘得基函数分子的总值 
        xia=xia*(xx-x1(b));      % 连乘得基函数分母的总值
    end                % 结束for循环
    a0(a)=shang/xia;    % 得到在a循环下的每一个值下基函数的值
end                %使用for循环遍历
L=a0*y';    % 进行矩阵运算（简化运算）（y为线性结构）




%% 正向化 

%封装一个mintomax函数（用于正向化）（单独的m文件）
function [posit_x] = mintomax(x)
    posit_x = max(x) - x;
end




%% 拟合

function NH=nh(x,y,x0)
n=sum(F);
lzc=find(F)-1;
A=zeros(n);
ssh=zeros(n,1);
for i=1:n
    for j=1:n
        A(i,j)=(x.^lzc(i))*(x.^lzc(j))';
    end
    ssh(i)=(x.^lzc(i))*y';
end
a=A'-1*ssh;
NH=0;
for k=1:n
    NH=NH+a(k)*(x0.^lzc(k));
end
end




%% 牛顿插值算法

%牛顿插值算法实现：给样本点和代插节点算插值的值
%函数格式代码，包含注释
function N=f(x,y,x0)   %定义N函数
f=y;         %将y赋值给f
jeh=x0;      %把x0赋值给jeh
oyxy=length(x);   %把x的长度赋值给oyxy
zl=zeros(oyxy-1,oyxy-1);    %预先生成相应大小的全零矩阵
for zyr=1:oyxy-1           %for循环遍历zyr
    for zbh=1:oyxy-zyr      %for循环遍历zbh
        zl(zbh,zyr)=(f(zbh+1)-f(zbh))/(x(zbh+zyr)-x(zbh));  %一个一个计算zl
    end         %结束第一个for循环
    f=zl(:,zyr);    % 将其定在j列，最后可以得到一个左上角矩阵
end         %结束第二个for循环
zzw=zl(1,:);     % zzw是zl的1行j列
zdm=zeros(1,oyxy-1);       %预先生成zdm全零矩阵
for zbh=1:oyxy-1    %for循环遍历zbh
    zyh=1;          %把1赋值给zyh
    for zyr=1:zbh     %for循环遍历zyr
        zyh=zyh*(jeh-x(zyr));   % 连乘得zyh尾部
    end           %结束第三个for循环
    zdm(zbh)=zyh;      %得到究极zdm
end         %结束最后一个循环
N=y(1)+zzw*zdm';   % N的矩阵运算（线性结构）




%% 根据不同类型的数据实现正向化

function [posit_x] = Pos(x, type)
    % 检查输入有效性
    if ~isvector(x)
        error('参数 x 必须是一个向量或数组.');
    end
    if ~ismember(type, [1, 2, 3])
        error('无此类指标，请检查 type 参数是否为 1, 2 或 3.');
    end

    switch type
        case 1  % 极小型
            posit_x = mintomax(x);  % 调用 mintomax 函数来正向化
        case 2  % 中间型
            best = input('请输入最佳的那一个值： ');
            posit_x = mintomax(x, best);  % 调用 mintomax 函数，并传入最佳值
        case 3  % 区间型
            a = input('请输入区间下限： ');
            b = input('请输入区间上限： ');
            posit_x = Inter2Max(x, a, b);  % 调用 Inter2Max 函数
        otherwise
            error('发生错误，请检查输入类型。');
    end
end




%% 三次样条插值

%按照时间段/三次样条插值
x=[1 2 3 5 6 7 8]
y=[25996 30106 30660 25687 25573 24393 24403]
a=4
y1=interp1(x,y,a,'spline')   %y1=28096

%按照时间段/拟合
x=[1 2 3 5 6 7 8]
y=[25996 30106 30660 25687 25573 24393 24403]
P=polyfit(x,y,3)
xi=4
yi=polyval(P,xi)   %yi=29112

%按照日期/三次样条插值
x=[1 2 3 5 6 7 8]
y=[30693 31565 24843 25587 28734 33669 20529]
a=4
y1=interp1(x,y,a,'spline')   %y1=23363

%按照日期/拟合
x=[1 2 3 5 6 7 8]
y=[30693 31565 24843 25587 28734 33669 20529]
P=polyfit(x,y,3)
xi=4
yi=polyval(P,xi)  %yi=26904

%异常值为240470，预计为多一个0，实际值为24047，最接近为法三
