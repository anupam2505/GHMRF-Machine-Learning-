
function GI=gaussianBlur(I,s)


h=fspecial('gaussian',ceil(s)*3+1,s);

GI=imfilter(I,h,'replicate');
