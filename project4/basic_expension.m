function z = basic_expension(x, P)
% [rows, cols] = size(x);
% z=ones(rows, 1);
z=[];
for p=0:P
  
    z_temp=x.^p;
    z=horzcat(z,z_temp);

end