function index=argmax(A)
    n=size(A,2);
    index=zeros(1,n);
    for i=1:n
        [~,index(i)] = max(A(:,i));
    end

end