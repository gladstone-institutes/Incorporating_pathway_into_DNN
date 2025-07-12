

library(Seurat)
library(biomaRt)
library(cluster)


pathwayN = switch(db,'wp'= wp_pathwaydb,
                  'reactome'= reactome_pathwaydb,
                  'kegg' = kegg_pathwaydb,
                  'pfocr' = pfocr_pathwaydb,
)




entrez = useMart(biomart = "ensembl", dataset = "hsapiens_gene_ensembl")
gene_id_to_entrez <- getBM(
  attributes = c("entrezgene_id", "hgnc_symbol"),
  filters = "hgnc_symbol",
  values = rownames(data.combined2),
  mart = entrez
)

types=c("Breast", "Ovarian", "Prostate", "Kidney")

sil_all_means=NULL
sil_pathway_means=NULL

num_genes=NULL

for(set_id in c(4)){
  
  index<-read.table(paste0("3CA_",db, "/hp",set_id,"/stepfoward_final_path_index_cvdi0"), sep=",", header=F)
  
  path<-read.table(pathwayN, sep=",")
  path=path[,unlist(index)+1]
  dim(path)
  path=path[rowSums(path)>0,]
  dim(path)
  
  geneNs=paste0(rownames(path))
  

  
  setwd(paste0("3CA_",db,"/hp",set_id,"/"))
  
  
  data.combined2.sub<-readRDS( paste0("hp",set_id,"_stepwise_seurat.rds"))



  # Randomly sample 100 cells if there are more
  selected_cells <- sample(Cells(data.combined2), 5000)
  # Create a new Seurat object with the sampled cells
  data.combined3 <- subset(data.combined2, cells = selected_cells)
  

  
  
  dp=DimPlot(object = data.combined3, reduction = 'umap',  group.by = 'cell_type')
  
  tiff(paste0("snn_all_gene_umap.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  
  
  subdata=subset(data.combined3, cell_type == types[set_id])
  counts <- GetAssayData(subdata, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  target=colnames(counts)
  
  subdata=subset(data.combined3, cell_type != types[set_id])
  counts <- GetAssayData(subdata, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  nontarget=colnames(counts)
  
  
  dp=DimPlot(data.combined3, label=T, group.by="cell_type", cells.highlight= list(target, nontarget), cols.highlight = c("black", "grey"), cols= "grey")
  
  tiff(paste0("snn_all_gene_", types[set_id],"_highlighted.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  
  
  dp=DimPlot(data.combined3, label=T, group.by="orig.ident")
  
  tiff(paste0("snn_all_gene_", types[set_id],"_sample.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  
  
  
  data.combined3 <- FindNeighbors(data.combined3, reduction = "pca", dims = 1:5)
  
  snn_matrix <- as.matrix(data.combined3@graphs$RNA_snn)
  snn_distance_matrix <- 1 - snn_matrix  # Assuming values between 0 and 1
  membershp0=matrix(0, nrow=ncol(snn_matrix), ncol=1)
  membershp0[data.combined3$cell_type==types[set_id]]=1
  

  
  distM=dist(snn_distance_matrix)
  distM=as.matrix(distM)
  
  
  sil_all <- silhouette(membershp0, distM)
  sil_all_summary=summary(sil_all)
  sil_all_mean=sil_all_summary$avg.width
  sil_all_mean
  sil_all_means=c(sil_all_means,sil_all_mean)
  
  
  
  # Randomly sample 
  selected_cells <- sample(Cells(data.combined2.sub), 5000)

  
  # Create a new Seurat object with the sampled cells
  data.combined3.sub <- subset(data.combined2.sub, cells = selected_cells)
  data.combined3.sub <- FindNeighbors(data.combined3.sub, reduction = "pca", dims = 1:5)
  
  subdata=subset(data.combined3.sub, cell_type == types[set_id])
  counts <- GetAssayData(subdata, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  target=colnames(counts)
  
  subdata=subset(data.combined3.sub, cell_type != types[set_id])
  counts <- GetAssayData(subdata, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  nontarget=colnames(counts)
  
  
  dp=DimPlot(data.combined3.sub, label=T, group.by="cell_type", cells.highlight= list(target, nontarget), cols.highlight = c("black", "grey"), cols= "grey")
  
  tiff(paste0("snn_stepwise_", types[set_id],"_highlighted.tiff"), width=1000, height=500)
  print(dp)
  dev.off()

  snn_matrix <- as.matrix(data.combined3.sub@graphs$RNA_snn)
  snn_distance_matrix <- 1 - snn_matrix  # Assuming values between 0 and 1
  
  distM=dist(snn_distance_matrix)

  distM=as.matrix(distM)
  
  membershp=matrix(0, nrow=ncol(snn_matrix), ncol=1)
  membershp[data.combined3.sub$cell_type==types[set_id]]=1
  
  
  sil_pathway=silhouette(membershp, distM)
  sil_pathway_summary=summary(sil_pathway)
  sil_pathway_mean=sil_pathway_summary$avg.width
  sil_pathway_means=c(sil_pathway_means,sil_pathway_mean)

  
  
  ##### original
  
  
  pathindex = read.csv(paste0('3CA_',db,'/hp',set_id,'/stepfoward_final_path_index_cvdi0'), header=F)
  pathindex = unlist(pathindex)
  index2<-read.csv(paste0("3CA_",db,"/hp",set_id,"/original_model_gene_importance_cvdi0"), sep=",", header=T)
  # python based pathway index starting from 0

  index2=index2[order(index2[,2], decreasing = T),]
  geneNs2=index2[1:length(geneNs),1]
  
  counts <- GetAssayData(data.combined2, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  
  rownames(counts)=gene_id_to_entrez[match(rownames(data.combined2), gene_id_to_entrez[,2]),1]
  
  counts.sub <- counts[as.character(geneNs2),]
  
  data.combined2.sub <- CreateSeuratObject(counts=counts.sub)
  
  data.combined2.sub=AddMetaData(object = data.combined2.sub, metadata = data.combined2@meta.data$cell_type, col.name = "cell_type")
  data.combined2.sub=AddMetaData(object = data.combined2.sub, metadata = data.combined2@meta.data$sample, col.name = "sample")
  
  
  selected_cells <- sample(Cells(data.combined2.sub), 5000)

  
  
  # Create a new Seurat object with the sampled cells
  data.combined3.sub <- subset(data.combined2.sub, cells = selected_cells)
  
  
  data.combined3.sub <- NormalizeData(data.combined3.sub)
  data.combined3.sub <- ScaleData(data.combined3.sub)
  data.combined3.sub <- FindVariableFeatures(data.combined3.sub)
  data.combined3.sub <- RunPCA(data.combined3.sub)
  data.combined3.sub <- FindNeighbors(data.combined3.sub, reduction = "pca", dims = 1:5)
  
  
  data.combined3.sub <- RunUMAP(object = data.combined3.sub, dims = 1:5)
  
  dp=DimPlot(object = data.combined3.sub, reduction = 'umap',  group.by = 'cell_type')
  
  setwd(paste0("3CA_",db,"/hp",set_id,"/"))
  
  tiff(paste0("snn_topSHAP_gene_umap.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  
  data.combined2#10540 features across 144926 samples
  data.combined3.sub#10540 features across 2431 samples
  
  subdata=subset(data.combined3.sub, cell_type == types[set_id])
  counts <- GetAssayData(subdata, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  target=colnames(counts)
  
  subdata=subset(data.combined3.sub, cell_type != types[set_id])
  counts <- GetAssayData(subdata, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  nontarget=colnames(counts)
  
  dp=DimPlot(data.combined3.sub, label=T, group.by="cell_type", cells.highlight= list(target, nontarget), cols.highlight = c("black", "grey"), cols= "grey")
  
  tiff(paste0("snn_topSHAP_gene_", types[set_id],"_highlighted.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  
  dp=DimPlot(data.combined3.sub, label=T, group.by="sample")
  
  tiff(paste0("snn_topSHAP_gene_", types[set_id],"_sample.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  

  
  snn_matrix <- as.matrix(data.combined3.sub@graphs$RNA_snn)
  snn_distance_matrix <- 1 - snn_matrix  # Assuming values between 0 and 1
  
  distM=dist(snn_distance_matrix)
  
  
  distM=as.matrix(distM)
  
  membershp=matrix(0, nrow=ncol(snn_matrix), ncol=1)
  membershp[data.combined3.sub$cell_type==types[set_id]]=1
  
  
  sil_pathway=silhouette(membershp, distM)
  sil_pathway_summary=summary(sil_pathway)
  sil_pathway_mean=sil_pathway_summary$avg.width
  sil_pathway_means=c(sil_pathway_means,sil_pathway_mean)
  

  
  ##### original + stepwise
  
  
  pathindex=read.csv(paste0('3CA_',db,'/hp',set_id,'/stepfoward_final_path_index_cvdi0'), header=F)
  pathindex=unlist(pathindex)
  index2<-read.csv(paste0("3CA_",db,"/hp",set_id,"/original_plus_pathways_model_gene_importance"), sep=",", header=T)
  # python based pathway index starting from 0
  pathway_is=c(grep("[^0-9]",index2[,1]))
  index2=index2[-pathway_is,]
  index2=index2[order(index2[,2], decreasing = T),]
  geneNs2=index2[1:length(geneNs),1]
  
 
  
  counts <- GetAssayData(data.combined2, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  
  rownames(counts)=gene_id_to_entrez[match(rownames(data.combined2), gene_id_to_entrez[,2]),1]
  
  counts.sub <- counts[union(geneNs, as.character(geneNs2)),]

  
  data.combined2.sub <- CreateSeuratObject(counts=counts.sub)
  
  
  data.combined2.sub=AddMetaData(object = data.combined2.sub, metadata = data.combined2@meta.data$cell_type, col.name = "cell_type")
  data.combined2.sub=AddMetaData(object = data.combined2.sub, metadata = data.combined2@meta.data$sample, col.name = "sample")
  
  
  selected_cells <- sample(Cells(data.combined2.sub), 5000)
  
  data.combined3.sub <- subset(data.combined2.sub, cells = selected_cells)


  
  data.combined3.sub <- NormalizeData(data.combined3.sub)
  data.combined3.sub <- ScaleData(data.combined3.sub)
  data.combined3.sub <- FindVariableFeatures(data.combined3.sub)
  data.combined3.sub <- RunPCA(data.combined3.sub)
  data.combined3.sub <- FindNeighbors(data.combined3.sub, reduction = "pca", dims = 1:5)
  
  
  data.combined3.sub <- RunUMAP(object = data.combined3.sub, dims = 1:5)
  
  
  dp=DimPlot(object = data.combined3.sub, reduction = 'umap',  group.by = 'cell_type')
  
  setwd(paste0("3CA_",db,"/hp",set_id,"/"))
  
  tiff(paste0("snn_stepwise_plus_original_topSHAP_umap.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  


  subdata=subset(data.combined3.sub, cell_type == types[set_id])
  counts <- GetAssayData(subdata, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  target=colnames(counts)
  
  subdata=subset(data.combined3.sub, cell_type != types[set_id])
  counts <- GetAssayData(subdata, slot="counts", assay="RNA")     #select genes expressed in at least 1% of cells
  nontarget=colnames(counts)
  
  dp=DimPlot(data.combined3.sub, label=T, group.by="cell_type", cells.highlight= list(target, nontarget), cols.highlight = c("black", "grey"), cols= "grey")
  
  tiff(paste0("snn_stepwise_plus_topSHAP_", types[set_id],"_highlighted.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  
  dp=DimPlot(data.combined3.sub, label=T, group.by="sample")
  
  tiff(paste0("snn_stepwise_plus_topSHAP_", types[set_id],"_sample.tiff"), width=1000, height=500)
  print(dp)
  dev.off()
  
  
  snn_matrix <- as.matrix(data.combined3.sub@graphs$RNA_snn)
  snn_distance_matrix <- 1 - snn_matrix  # Assuming values between 0 and 1
  
  distM=dist(snn_distance_matrix)
  

  
  membershp=matrix(0, nrow=ncol(snn_matrix), ncol=1)
  membershp[data.combined3.sub$cell_type==types[set_id]]=1
  
  

  
  sil_pathway=silhouette(membershp, distM)
  sil_pathway_summary=summary(sil_pathway)
  sil_pathway_mean=sil_pathway_summary$avg.width
  sil_pathway_means=c(sil_pathway_means,sil_pathway_mean)
  
}

