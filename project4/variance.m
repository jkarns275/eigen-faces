function  p=variance(eigvalue,th, pp)
    vsum=sum(eigvalue);
    S = 0;
    p = 0;
    if pp==1
        s_v=zeros(1,size(eigvalue,1));
        for i = 1: size(eigvalue, 1)
            S=(S+eigvalue(i)/vsum);
            s_v(i)=S; 
        end
        figure();
        plot(s_v);
        xlim([0,350]);
        for p = 1:size(eigvalue, 1)
            if s_v(p) >= th
                break
            end
        end
    end
    if pp==0
        
        while S/vsum<th
            p=p+1;
            S=S+eigvalue(p);
        end
    end
end