function x = XFromFzEta(fz, eta, eVec, e, anchorVertexIndex)

% tic;
fzbar = eta./fz;

% f_on_edges = 	0.5 *( eVec .* (fz_t_start + fz_t_end) + ... %Phi
%                  conj( eVec .* (fzbb_t_start + fzbb_t_end)) ); %conj(Psy)

f_on_edges = 	0.5 *( eVec .* sum(fz(e), 2) + conj( eVec .* sum(fzbar(e), 2)) );

f_on_anchor_vertex = complex(0, 0);
x = treeCumSum(anchorVertexIndex, f_on_anchor_vertex, gather( f_on_edges ), e(:,1), e(:,2));
% toc