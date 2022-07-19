function gX = LogFzFromTreeRoute(fz,e,anchorVertexIndex)

gdifFromFz = complex( log( abs( fz(e(:,2),:)./fz(e(:,1),:) ) ) , angle( fz(e(:,2),:)./fz(e(:,1),:) ) );

gX = log(fz(anchorVertexIndex))+treeCumSum(anchorVertexIndex, complex(0,0), gather(gdifFromFz), e(:,1), e(:,2));

end

