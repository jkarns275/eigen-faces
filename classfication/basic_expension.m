function z = basic_expension(x, P)
[rows, cols] = size(x);
z=ones(rows, 1);
for p=1:P
  
    z_temp=x.^p;
    z=horzcat(z,z_temp);

end