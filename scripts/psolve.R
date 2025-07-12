
library(lpSolve)


dt=dt[,colSums(dt)<101&colSums(dt)>4]
dt=dt[rowSums(dt)>0,]

f.obj <-  rep(1,ncol(dt))  # Numeric vector of coefficients of objective function
f.dir <- rep(">=",nrow(dt)) # the constraint "directions" by row
f.rhs <- rep(1,nrow(dt))    # the inequality values by row (require all items to be present)

sol=lp ("min", f.obj, dt, f.dir, f.rhs)$solution
length(sol)==dim(dt)[2]
sum(sol>0)
dt=dt[,sol>0 ]
sort(colSums(dt))

