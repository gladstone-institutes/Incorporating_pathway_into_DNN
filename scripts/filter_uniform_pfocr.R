library(cluster)
set.seed(6116)

dt<-read.delim("PFOCR24_UMAPCoordinates.tsv")

clusters <- kmeans(dt[,c("x","y")], centers = N)

# Initialize a vector to store the selected points
selected_points <- data.frame(x = numeric(N), y = numeric(N))

# Loop through each cluster to find the point closest to the cluster center
for (i in 1:N) {
  # Get the points in the i-th cluster
  cluster_points <- dt[clusters$cluster == i,c("x", "y") ]
  
  # Calculate distances to the cluster center
  distances <- apply(cluster_points, 1, function(point) {
    sqrt(sum((point - clusters$centers[i, ])^2))
  })
  
  # Select the point closest to the cluster center
  selected_points[i, ] <- cluster_points[which.min(distances), ]
}

result = apply(selected_points, 1, function(point) {
  dt[which(dt[,"x"]==point[1] & dt[,"y"]==point[2]),]
})
result2 = do.call(rbind, result)
