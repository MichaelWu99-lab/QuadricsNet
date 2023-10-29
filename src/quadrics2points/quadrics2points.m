function fv = quadrics2points(data_path)

    data = importdata(data_path);
    q = data(1:10);
    mesh_size = reshape(data(11:16),2,3)';
    res= data(17:19);
    error = data(20);
    xlow = mesh_size(1,1);
    xhigh = mesh_size(1,2);
    ylow = mesh_size(2,1);
    yhigh = mesh_size(2,2);
    zlow = mesh_size(3,1);
    zhigh = mesh_size(3,2);
    x = xlow:res(1):xhigh;
    y = ylow:res(2):yhigh;
    z = zlow:res(3):zhigh;
    [X, Y, Z]=meshgrid(x, y, z);
    I = ones(size(X));
    
    F = q(1) .* X.^2 + q(2) .* Y.^2 + q(3) .* Z.^2 + ...
        q(4) .* 2 .* X .* Y + q(5) .* 2 .* X .* Z + q(6) .* 2 .* Y .* Z + ...
        q(7) .* 2 .* X + q(8) .* 2 .* Y + q(9) .* 2 .* Z + q(10) .* I;

    fv = isosurface(X, Y, Z, F, error);
end