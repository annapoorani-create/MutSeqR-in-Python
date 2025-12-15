import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="ECS VCF Analysis", layout="wide")
st.header("ECS VCF Analysis")

df = pd.read_csv("synthetic_large_mutation_dataset.csv")

with st.expander("View the dataset head:"):
    st.dataframe(df.head(15))
with st.expander("View column descriptions:"):
    st.markdown("""
IMPORTANT: This is a MADE UP dataset solely for the purpose of having something to write these algorithms on. It had nothing to do with any real organism's DNA.

The first column is chromosome pair all this data was gathered from. In this case, all the data is from chromosome pair 1.

The next column is position. This represents the starting nucelotide # where this mutation occurred. The table only represents SNVs, or mutations where once base was switched for another, so the 'end' column, where the mutation ended, is just pos+1.

REF shows the nucleotide that would occur in the given position in a normal oragnism. ALT shows the nucleotide that was read at least once when scanning through the genome at that position.

VD is essentially a higher-quality ALT_DEPTH; however, I've used ALT_DEPTH for the code to make it more readable. See below for an explanantion on ALT_DEPTH.

CTX is the exact mutation that occured. For example, G[A>C]T in the CTX column means that there were supposed to be three nucleotides, in order, that were GAT. However, a mutation occurred that made it show up in some of the reads as GCT.

SAMPLE tells us what organism/dosage level this data is from. For example, there might be a sample for low radiation, or a sample all gathered from a particular mouse. This is the column I have used to group by for calculate_mf, since there is no dosage column and thus no other column that would be useful to group by.

GT is the genotype. In a chromosome pair, there are two alleles (versions of the same gene) and it's possible that one has the mutation and the other doesn't (heterozygous), or that both have the mutation (homozygous). 1/1 means homozygous, and 0/1 means heterozygous.

REF_DEPTH is the number of reads of the genome in which the reference allele (i.e. not a mutation) was obseved. A read is a scan through the DNA, and different reads often happen on different cells (all cells contain the same DNA).

ALT_DEPTH is the number of reads of the genome in which the mutation was observed.

DP is the total depth (sum of REF and ALT DEPTHS) - the total number of times a certain position was read over.
    """)

# calculate mf function
def calculate_mf(df, cols_to_group="SAMPLE", subtype_resolution="base_12"):
    df = df.copy()

    if subtype_resolution == "base_96":
        df["normalized_ref"] = df["CTX"].str[0] + df["CTX"].str[2] + df["CTX"].str[6]
    if subtype_resolution == "base_12":
        df["normalized_ref"] = df["CTX"].str[2]
        df["CTX"] = df["CTX"].str[2] + df["CTX"].str[3] + df["CTX"].str[4]

    df["alt_depth_min"] = (df["ALT_DEPTH"] > 0).astype(int)

    group_cols = [cols_to_group, "CTX"]
    counts = (
        df.groupby(group_cols, as_index=False)
        .agg(sum_max=("ALT_DEPTH", "sum"), sum_min=("alt_depth_min", "sum"))
    )

    depth_group = df.groupby(cols_to_group, as_index=False).agg(group_depth=("DP", "sum"))
    depth_subtype = (
        df.groupby([cols_to_group, "normalized_ref"], as_index=False)
        .agg(subtype_depth=("DP", "sum"))
    )

    if subtype_resolution == "base_96":
        counts["normalized_ref"] = counts["CTX"].str[0] + counts["CTX"].str[2] + counts["CTX"].str[6]
    if subtype_resolution == "base_12":
        counts["normalized_ref"] = counts["CTX"].str[0]

    mf = counts.merge(depth_group, on=cols_to_group, how="left")
    mf = mf.merge(depth_subtype, on=[cols_to_group, "normalized_ref"], how="left")

    mf["total_sum_min"] = mf.groupby(cols_to_group)["sum_min"].transform("sum")
    mf["total_sum_max"] = mf.groupby(cols_to_group)["sum_max"].transform("sum")
    mf["proportion_min"] = mf["sum_min"] / mf["total_sum_min"]
    mf["proportion_max"] = mf["sum_max"] / mf["total_sum_max"]

    mf["D_s_i"] = mf["subtype_depth"]
    M_s_i = df.groupby([cols_to_group, "CTX"], as_index=False).agg(M_s_i=("ALT_DEPTH", "sum"))

    mf = mf.merge(M_s_i, on=[cols_to_group, "CTX"], how="left")
    mf["Msi/Dsi"] = mf["M_s_i"] / mf["D_s_i"]

    denominator_sum = mf.groupby([cols_to_group], as_index=False).agg(denominator_sum=("Msi/Dsi", "sum"))
    mf = mf.merge(denominator_sum, on=[cols_to_group], how="left")
    mf["normalized_proportion"] = mf["Msi/Dsi"] / mf["denominator_sum"]

    mf = mf.drop(columns=["total_sum_min", "total_sum_max", "D_s_i", "M_s_i", "Msi/Dsi", "denominator_sum"])
    return mf

# --- UI controls ---
subtype = st.radio("Subtype resolution", ["base_12", "base_96"], horizontal=True)

if st.button("Run Calculate_MF"):
    st.session_state["mldf"] = calculate_mf(df, subtype_resolution=subtype)

# --- show results if available ---
if "mldf" in st.session_state:
    mldf = st.session_state["mldf"]

    st.subheader("Calculated MF table head:")
    st.dataframe(mldf.head(15))
    with st.expander("What is this showing?"):
        st.markdown("""Each row in this table represents all the mutations in a certain sample that were of the same type (based on subtype). For example, at base 96, the first row might encapsulate info about all mutations of the type A[A>C]A in sample 1, or at base 12, all mutations of the type A>C. That's what the first two columns in a row tell us: the sample, and the 'CTX', or exact mutation.
        
Next, we have the 'sum_max' and 'sum min' columns. When we look at our original VCF, there are typically many rows that have the same type of mutation, just at different positions in the genome. For each of these, we have read depths: how many times that position was read (DP), and how many times in those reads a mutation was detected (ALT_DEPTH). Because reads are usually conducted in different cells, the mutation could have ocurred independently in each cell, or could have ocurred independently in just one cell, then passed to all the cells via replication. So, we assume the actual number of independently caused mutations is somewhere in-between these two, and call the former method of counting 'max' counting and the latter method 'min' counting. 
        
For max counting, we just sum all the alt depths for a certain mutation - these are all occurences in different cells, most likely, so we pretend each event occurred independently. For min counting, we add up the total number of rows in which this particular mutation ocurred in this particular sample - that is, each mutation only ocurred once, and all the extra reads were found only because of cell replication. 
        
The next column is normalized ref. This is the context in which the mutation could have ocurred, and comes in two resolutions: 96 or 12. If we have the mutation A[A>C]A in base 96, where an A became a C, the normalized ref is AAA. This is just telling us that spots on the reference genome with a sequential AAA could have accumulated this mutation. In base 12, where we have a mutation like A>C, the normalized ref is just A - any A on the reference genome could have accumulated this mutation.
        
Group depth is the total depth across the entire sample. It is the total number of reads done anywhere on the genome.\

Subtype depth is the total depth summed across all elements of the sample that shared the same normalized ref. It represents the total number of mutation opportunities for any given specific mutation.
        
Proportion min and proportion max are the raw proportions that a certain mutation represented in the total set of mutations (in each sample). For proportion min, with sum together all the min counts of every mutation in a given sample; then, for each row, we take its sum_min for the mutation it represents and divide by the total sum of all the min counts. For example, if, by min counting, we observed 400 mutations on the genome, and 20 of them were of type A>C (i.e. type A>C was found in 20 different positions) then the min_proportion for A>C mutations is 0.05. The same is done for max, except instead of using min counting on the numerator and denominator, max counting is used.
        
Normalized proportion represents the probability that a specific type of mutation occurred, GIVEN that a mutation occurred at all. First, we must compute how likely any given mutation (call it mutation M) is to occur. The denominator, of course, is the number of opportunities M had to occur - the sum of the total depths of normalized ref. The numerator is the sum of the alt depths of the exact mutation M - how many times the mutation actually occurred. Then, the rate of M occurring is (the sum of the alt depths of M)/(the sum of the total depths of the positions where it could have occurred). Now, we compute this fraction for every kind of mutation. The sum of these fractions becomes the factor we normalize by to get the likelihood of M happening with respect to all the other possible mutations - it's the denominator. Then, for each row, its fraction (rate) is the numerator we divide by this denominator to get the normalized proportion. 
        
For example, say there are only 2 mutations, M and N. We observed M in 1 read, and we observed M's normalized reference (the positions where it could have occurred) 4 times. We observed N in 1 read also, but only observed its normalized reference 2 times. The rate of occurrence for M is 1/4, but the rate of occurrence for N is 1/2. Given that at least 1 of these mutations happened, N is twice as likely as M to have occurred, so its probability is 2/3 - with could have also been computed as (1/2)/(1/2+1/4). This concept is what we aply to normalized proportion, only across many more specific types of mutations.
        """)

    st.write("Rows:", len(mldf), "Columns:", len(mldf.columns))

    number_of_facets = st.radio("Display the top ____ normalized references in the heatmaps (note that there are only 4 total references for base_12 resolution):",['2','4','8','12','16']) 
    st.session_state["number_of_facets"] = int(number_of_facets)
    if st.button("Generate heatmap"):
        mldf = st.session_state["mldf"]
        x = st.session_state["number_of_facets"]
        import math
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        facets = sorted(mldf["normalized_ref"].unique())[:x]
        
        ncols = 4
        nrows = math.ceil(len(facets) / ncols)
        
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4 * ncols, 4 * nrows),
            sharey=True
        )
        
        axes = axes.flatten()
        
        for ax, normalized_ref in zip(axes, facets):
            subset = mldf[mldf["normalized_ref"] == normalized_ref]
        
            heatmap_df = subset.pivot_table(
                index="SAMPLE",
                columns="CTX",
                values="proportion_min",
                aggfunc="mean"
            )
        
            sns.heatmap(
                heatmap_df,
                cmap="viridis",
                ax=ax,
                cbar=False   # recommended for multi-panel plots
            )
        
            ax.set_title(f"normalized_ref = {normalized_ref}")
            ax.set_xlabel("Mutation context")
        
        # Turn off unused axes
        for ax in axes[len(facets):]:
            ax.axis("off")
        
        axes[0].set_ylabel("Sample")
        plt.tight_layout()
        st.pyplot(fig)


    if st.button("Generate PCA plot"):
        features = ["sum_max", "sum_min", "subtype_depth", "proportion_min", "proportion_max", "normalized_proportion"]
        X = mldf[features].dropna()

        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_pca[:, 0], X_pca[:, 1],
                   c=mldf.loc[X.index, "SAMPLE"].astype("category").cat.codes,
                   alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA of Mutation Proportion Features")

        st.pyplot(fig)
        with st.expander("What is this plot showing?"):
            st.markdown("""PCA, or principal component analysis, is an unsupervised learning model that shows us whether our data contains patterns based on a features we tell it to examine. To make for easy visual analysis, PCA compresses the many features into just two 'principal components' that can be plotted on the x and y axes. These principal components are essentially just linear combinations that combine several features into a single output number. The algorithm order the principal components and see which describe the most spread in the data (i.e., show the most variance). This allows it to highlight the differences best, since lines with a lot of variance usually contain the most information.
            
The top two principal components are what go on the x and y axes, since these show the range of data in the broadest way possible. In data that has correlations that arise when multiple features are considered together, this creates clusters of points that indicate there are patterns we might be able to use supervised models to analyze.
            
The features we used were "sum_max", ."sum_min", "subtype_depth", "proportion_min", "proportion_max", and "normalized_proportion". In this case, there appears to be no grouping based on sample (I added in these labels after PCA, to make clear there was no grouping); all three samples are mixed together. This indicates that these features are not a good predictor of sample. This data is not real, so we do not know if this is true in general.
            """)

        st.write("Explained variance ratio:", pca.explained_variance_ratio_)
        st.write("Total variance explained (PC1+PC2):", float(pca.explained_variance_ratio_.sum()))
        with st.expander("What are these metrics?"):
            st.text("These metrics show us how much of the total variance in the dataset PCA managed to capture. We want to capture as much as possible with just two axes to have the best chance of spotting any patterns.")

   
