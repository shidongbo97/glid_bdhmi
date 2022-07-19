function x = mywarning(x, v)

% i = x>v+eps('single');
i = x>v*1.01;
if any(i)
    if isscalar(v)
        warning('max: %f > ref: %f, %.2f%%!\n', max(x), v, (max(x)-v)/v*100); 
    else
        i = find(i);
        [~, j] = max(x(i)./v(i));
        i = i(j);
        warning('max: %f > ref: %f, %.2f%%!\n', [x(i) v(i) (x(i)-v(i))./v(i)*100]'); 
    end
end