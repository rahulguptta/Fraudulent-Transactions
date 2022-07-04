<h1>Fradulent Transactions <br> (Classification of Imbalanced Dataset)</h1>
<h2 id="introduction">Introduction</h2>
<ul>
<li><p>FRAUDULENT TRANSACTIONS is a case of imbalanced dataset. Imbalanced dataset are the dataset where the presence of once class is very high compared to the second class. For example spame data for email.</p></li>
<li><p>The issue with imbalanced data is that most of the classification algorithms are designed for approximately equal sample of both classes.</p></li>
<li><p>For example if we take case of credit card transanctions, maximum 2-3 out of 100 transactions will be fraud. In this case if I do nothing still the accuracy for zero lable is 96% which is very high.</p></li>
<li><p>Some classifiers are created in this project for FRAUDULENT TRANSACTIONS and compared. The dataset used for this project is downloaded from <a href="https://www.kaggle.com/datasets/vardhansiramdasu/fraudulent-transactions-prediction">Kaggle</a>.</p></li>
<li><p>Outline of the Project is</p>
<ul>
<li>General overview of the data</li>
<li>Visualization</li>
<li>Issues with the Default classifier</li>
<li>Classification using resampling
<ul>
<li>Resampling</li>
<li>Visualization</li>
<li>Model pipeline</li>
<li>Classificassion</li>
<li>Results</li>
</ul></li>
<li>Conclusion</li>
</ul></li>
</ul>
<section id="general-overview-of-the-data" class="cell markdown" id="oAhetDPLsXi9">
<h1>General overview of the data</h1>
</section>
<div class="cell code" data-execution_count="1" id="L0e1Q8PjIFmz">
<div class="sourceCode" id="cb1"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Importing the required Libraries</span></span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> numpy  <span class="im">as</span> np</span>
<span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> pandas <span class="im">as</span> pd</span>
<span id="cb1-4"><a href="#cb1-4" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> matplotlib.pyplot <span class="im">as</span> plt</span>
<span id="cb1-5"><a href="#cb1-5" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> seaborn <span class="im">as</span> sns</span>
<span id="cb1-6"><a href="#cb1-6" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> warnings</span>
<span id="cb1-7"><a href="#cb1-7" aria-hidden="true" tabindex="-1"></a>warnings.filterwarnings(<span class="st">&#39;ignore&#39;</span>)</span></code></pre></div>
</div>

<div class="cell code" data-execution_count="3" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="6KJxCRklsZd1" data-outputId="4fcbed2c-ccc8-4aa1-c92a-737c620d9d62">
<div class="sourceCode" id="cb4"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb4-1"><a href="#cb4-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Importing the data and general information</span></span>
<span id="cb4-2"><a href="#cb4-2" aria-hidden="true" tabindex="-1"></a>df <span class="op">=</span> pd.read_csv(<span class="st">&#39;Fraud.csv&#39;</span>)</span>
<span id="cb4-3"><a href="#cb4-3" aria-hidden="true" tabindex="-1"></a>df.info()</span></code></pre></div>
<div class="output stream stdout">
<pre><code>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 6362620 entries, 0 to 6362619
Data columns (total 11 columns):
 #   Column          Dtype  
---  ------          -----  
 0   step            int64  
 1   type            object 
 2   amount          float64
 3   nameOrig        object 
 4   oldbalanceOrg   float64
 5   newbalanceOrig  float64
 6   nameDest        object 
 7   oldbalanceDest  float64
 8   newbalanceDest  float64
 9   isFraud         int64  
 10  isFlaggedFraud  int64  
dtypes: float64(5), int64(3), object(3)
memory usage: 534.0+ MB
</code></pre>
</div>
</div>
<div class="cell code" data-execution_count="5" data-colab="{&quot;height&quot;:250,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="N4ZPaiQc1g0V" data-outputId="94a782e9-0efc-42bd-ed70-00a96a18e184">
<div class="sourceCode" id="cb6"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb6-1"><a href="#cb6-1" aria-hidden="true" tabindex="-1"></a><span class="co"># general view</span></span>
<span id="cb6-2"><a href="#cb6-2" aria-hidden="true" tabindex="-1"></a>df.head(<span class="dv">5</span>)</span></code></pre></div>
<div class="output execute_result" data-execution_count="5">

  <div id="df-95163111-618c-4c00-bbdd-c4f15554a89a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>isFlaggedFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>9839.64</td>
      <td>C1231006815</td>
      <td>170136.0</td>
      <td>160296.36</td>
      <td>M1979787155</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>1864.28</td>
      <td>C1666544295</td>
      <td>21249.0</td>
      <td>19384.72</td>
      <td>M2044282225</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>181.00</td>
      <td>C1305486145</td>
      <td>181.0</td>
      <td>0.00</td>
      <td>C553264065</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>181.00</td>
      <td>C840083671</td>
      <td>181.0</td>
      <td>0.00</td>
      <td>C38997010</td>
      <td>21182.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>11668.14</td>
      <td>C2048537720</td>
      <td>41554.0</td>
      <td>29885.86</td>
      <td>M1230701703</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-95163111-618c-4c00-bbdd-c4f15554a89a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-95163111-618c-4c00-bbdd-c4f15554a89a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-95163111-618c-4c00-bbdd-c4f15554a89a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
</div>
</div>
<div class="cell code" data-execution_count="6" data-colab="{&quot;height&quot;:399,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="CNfZZNC11SnF" data-outputId="9f98fd3c-0dc7-4ecc-8d0d-23f894e097b6">
<div class="sourceCode" id="cb7"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb7-1"><a href="#cb7-1" aria-hidden="true" tabindex="-1"></a><span class="co"># General view of the data lables</span></span>
<span id="cb7-2"><a href="#cb7-2" aria-hidden="true" tabindex="-1"></a>plt.style.use(<span class="st">&#39;dark_background&#39;</span>)</span>
<span id="cb7-3"><a href="#cb7-3" aria-hidden="true" tabindex="-1"></a>fig <span class="op">=</span> plt.gcf()</span>
<span id="cb7-4"><a href="#cb7-4" aria-hidden="true" tabindex="-1"></a>fig.set_size_inches(<span class="dv">10</span>, <span class="dv">6</span>)</span>
<span id="cb7-5"><a href="#cb7-5" aria-hidden="true" tabindex="-1"></a>sns.scatterplot(x <span class="op">=</span> <span class="st">&#39;oldbalanceOrg&#39;</span>, y <span class="op">=</span> <span class="st">&#39;newbalanceDest&#39;</span>, data <span class="op">=</span> df[df[<span class="st">&#39;isFraud&#39;</span>] <span class="op">==</span> <span class="dv">1</span>], color <span class="op">=</span> <span class="st">&#39;red&#39;</span>, label <span class="op">=</span> <span class="st">&#39;Fraud&#39;</span>, marker <span class="op">=</span> <span class="st">&#39;*&#39;</span>)<span class="op">;</span></span>
<span id="cb7-6"><a href="#cb7-6" aria-hidden="true" tabindex="-1"></a>sns.scatterplot(x <span class="op">=</span> <span class="st">&#39;oldbalanceOrg&#39;</span>, y <span class="op">=</span> <span class="st">&#39;newbalanceDest&#39;</span>, data <span class="op">=</span> df[df[<span class="st">&#39;isFraud&#39;</span>] <span class="op">==</span> <span class="dv">0</span>], color <span class="op">=</span> <span class="st">&#39;green&#39;</span>, label <span class="op">=</span> <span class="st">&#39;non-Fraud&#39;</span>, marker <span class="op">=</span> <span class="st">&#39;+&#39;</span>)<span class="op">;</span></span>
<span id="cb7-7"><a href="#cb7-7" aria-hidden="true" tabindex="-1"></a>plt.legend()</span>
<span id="cb7-8"><a href="#cb7-8" aria-hidden="true" tabindex="-1"></a>plt.show()</span></code></pre></div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/530ac2f91943b33b57728d014323b3e5a3e87205.png" /></p>
</div>
</div>
<div class="cell code" data-execution_count="7" data-colab="{&quot;height&quot;:325,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="YGWNjmdGvTfc" data-outputId="dd157d2d-d30d-4706-b0ef-42da1e1f2221">
<div class="sourceCode" id="cb8"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb8-1"><a href="#cb8-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Transaction labels </span></span>
<span id="cb8-2"><a href="#cb8-2" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">&#39;Number of fraud Transactions = &#39;</span>, df[df[<span class="st">&#39;isFraud&#39;</span>] <span class="op">==</span> <span class="dv">1</span>].shape[<span class="dv">0</span>])</span>
<span id="cb8-3"><a href="#cb8-3" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">&#39;Number of non-fraud Transactions = &#39;</span>, df[df[<span class="st">&#39;isFraud&#39;</span>] <span class="op">==</span> <span class="dv">0</span>].shape[<span class="dv">0</span>])</span>
<span id="cb8-4"><a href="#cb8-4" aria-hidden="true" tabindex="-1"></a>sns.countplot(x <span class="op">=</span> <span class="st">&#39;isFraud&#39;</span>, data <span class="op">=</span> df)<span class="op">;</span></span></code></pre></div>
<div class="output stream stdout">
<pre><code>Number of fraud Transactions =  8213
Number of non-fraud Transactions =  6354407
</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/527d74545f11f06041af7a0972d33fd30312db93.png" /></p>
</div>
</div>
<div class="cell markdown" id="fMe0ywyk1_Ys">
<ul>
<li>Here we can see that out 6362620 transactions, only 8213 (around 0.13 %) are fraud. If we just do the count plot, the fraud transaction is not even visible.</li>
<li>This is clearly an example of imbalanced dataset.</li>
<li>The data has 11 columns in which isFraud is the target column. In remaining columns, nameOrig and nameDest are not relevent features as they are the name of id of the costomures.</li>
<li>Column isFlagFraud contains 1, for all the transaction which are more than 200 dollars otherwise 0. Again this is not relevant for us.</li>
</ul>
</div>
<div class="cell code" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="ARDYxR66xSN8" data-outputId="3ab03cf6-8e00-4403-9316-281a447f79dd">
<div class="sourceCode" id="cb10"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb10-1"><a href="#cb10-1" aria-hidden="true" tabindex="-1"></a><span class="co"># checking for nans</span></span>
<span id="cb10-2"><a href="#cb10-2" aria-hidden="true" tabindex="-1"></a>df.isna().<span class="bu">sum</span>()</span></code></pre></div>
<div class="output execute_result" data-execution_count="7">
<pre><code>step              0
type              0
amount            0
nameOrig          0
oldbalanceOrg     0
newbalanceOrig    0
nameDest          0
oldbalanceDest    0
newbalanceDest    0
isFraud           0
isFlaggedFraud    0
dtype: int64</code></pre>
</div>
</div>
<div class="cell markdown" id="H-wKLcmcxguk">
<ul>
<li>It is clear that all the features have the correct data type with zero nans.</li>
<li>As we can see there are 6362620 samples in this data which is a quite high number. For simplicity we will use some randomaly generated samples.</li>
</ul>
</div>
<section id="visualization" class="cell markdown" id="Cnn6r_O1zKYL">
<h1>Visualization</h1>
</section>
<div class="cell code" id="UEsK_hxG3J7Z">
<div class="sourceCode" id="cb12"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb12-1"><a href="#cb12-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Getting a sample of 5000 and saving it for future reference</span></span>
<span id="cb12-2"><a href="#cb12-2" aria-hidden="true" tabindex="-1"></a>df_vis<span class="op">=</span> df.sample(<span class="dv">5000</span>, random_state <span class="op">=</span> <span class="dv">0</span>)</span>
<span id="cb12-3"><a href="#cb12-3" aria-hidden="true" tabindex="-1"></a>df_vis.to_csv(<span class="st">&#39;df_vis.csv&#39;</span>)</span></code></pre></div>
</div>
<div class="cell code" id="-mCdPrG2DVCM">
<div class="sourceCode" id="cb13"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb13-1"><a href="#cb13-1" aria-hidden="true" tabindex="-1"></a>df_vis <span class="op">=</span> pd.read_csv(<span class="st">&#39;df_vis.csv&#39;</span>)</span></code></pre></div>
</div>
<section id="does-the-fraud-transactions-happen-for-high-amount-if-yes-then-which-mode-is-being-used" class="cell markdown" id="xHoPgqXXzOqh">
<h2>Does the fraud transactions happen for high amount? If yes then which mode is being used?</h2>
</section>
<div class="cell code" data-colab="{&quot;height&quot;:291,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="xxT1VAkYzgrz" data-outputId="5a709115-cca6-4120-b2eb-f56c0a2aa867">
<div class="sourceCode" id="cb14"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb14-1"><a href="#cb14-1" aria-hidden="true" tabindex="-1"></a>sns.swarmplot(x <span class="op">=</span> <span class="st">&#39;type&#39;</span>, y <span class="op">=</span> <span class="st">&#39;amount&#39;</span>, hue <span class="op">=</span> <span class="st">&#39;isFraud&#39;</span>, data <span class="op">=</span> df_vis)<span class="op">;</span></span></code></pre></div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/e9a6705ff785b8c64825161b7b46d0a52ef04fee.png" /></p>
</div>
</div>
<div class="cell markdown" id="xILgeSXK1uxI">
<ul>
<li>Commenting about the transactions is hard here.</li>
<li>Other thing to notice is that most of the transaction are being transfered specially high amount transactions.</li>
<li>It is also intutive that the fraud transactions are mostly done by cashind out the money.</li>
</ul>
</div>
<section id="does-the-balance-in-acount-lead-towards-fraud-transactions-if-yes-then-how-much-amount" class="cell markdown" id="XjPvqCJK3oUQ">
<h2>Does the balance in acount lead towards fraud transactions? If yes then how much amount?</h2>
</section>
<div class="cell code" data-colab="{&quot;height&quot;:290,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="hk4wdbcz4C5A" data-outputId="d4d77fb3-5a48-43d3-ee73-981b23f8b884">
<div class="sourceCode" id="cb15"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb15-1"><a href="#cb15-1" aria-hidden="true" tabindex="-1"></a>sns.scatterplot(x <span class="op">=</span> <span class="st">&#39;oldbalanceOrg&#39;</span>, y <span class="op">=</span> <span class="st">&#39;amount&#39;</span>, hue <span class="op">=</span> <span class="st">&#39;isFraud&#39;</span>, data <span class="op">=</span> df_vis)<span class="op">;</span></span></code></pre></div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/cfda7cad26c2b42842d42a45796352d3f224d151.png" /></p>
</div>
</div>
<div class="cell markdown" id="D_raWRcV4SKr">
<ul>
<li>From the above graph it is clear that Fraud does not happen only for customers who have high balance in the account.</li>
</ul>
</div>
<section id="skewness-in-the-data" class="cell markdown" id="VUooNyim9dWt">
<h2>Skewness in the data</h2>
</section>
<div class="cell code" data-colab="{&quot;height&quot;:1000,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="fHKY6MkH9iG2" data-outputId="51988879-9975-4a69-d8d6-a7e007a293b3">
<div class="sourceCode" id="cb16"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb16-1"><a href="#cb16-1" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> col <span class="kw">in</span> [<span class="st">&#39;amount&#39;</span>,<span class="st">&#39;oldbalanceOrg&#39;</span>, <span class="st">&#39;newbalanceOrig&#39;</span>, <span class="st">&#39;oldbalanceDest&#39;</span>, <span class="st">&#39;newbalanceDest&#39;</span>]:</span>
<span id="cb16-2"><a href="#cb16-2" aria-hidden="true" tabindex="-1"></a>  sns.boxenplot(x<span class="op">=</span>col, data <span class="op">=</span> df_vis)<span class="op">;</span></span>
<span id="cb16-3"><a href="#cb16-3" aria-hidden="true" tabindex="-1"></a>  plt.show()</span>
<span id="cb16-4"><a href="#cb16-4" aria-hidden="true" tabindex="-1"></a>  <span class="bu">print</span>(col,<span class="st">&#39;</span><span class="ch">\n</span><span class="st">&#39;</span>)</span>
<span id="cb16-5"><a href="#cb16-5" aria-hidden="true" tabindex="-1"></a> </span></code></pre></div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/80ad5484ce04a3e19d751fbdb2c80a9be0ea3765.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>amount 

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/1c48071123110be4c6a8c1a714378e64d0148589.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>oldbalanceOrg 

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/405e970ec19ed61d9d80723ae94fffa1c986644d.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>newbalanceOrig 

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/0843f4f19eced5e63f0b17bae2651dbb88a45b31.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>oldbalanceDest 

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/34911f19a05257fe73c7dde7c47f64dd214c5d28.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>newbalanceDest 

</code></pre>
</div>
</div>
<div class="cell markdown" id="1N2dRWz_-JGb">
<ul>
<li>All the numeric features are highly right skewed and log transformation of the data will be required.</li>
<li>It is also fact that most of the accounts have less amount of money and it can be seen from the pictures.</li>
</ul>
</div>
<section id="issues-with-default-classifier-for-imbalanced-data" class="cell markdown" id="0t4-hOrj68e8">
<h1>Issues with default classifier for imbalanced data</h1>
</section>
<div class="cell markdown" id="CLy4P8AM4x0i">
<p>Let us create a some classifier to see the performance without changing anything.</p>
</div>
<section id="default-classfiers" class="cell markdown" id="vvZ09KOd7E2m">
<h2>Default Classfiers</h2>
</section>
<div class="cell code" data-execution_count="8" id="sv8WJjd87EMG">
<div class="sourceCode" id="cb22"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb22-1"><a href="#cb22-1" aria-hidden="true" tabindex="-1"></a><span class="co"># copy of data to check the performance</span></span>
<span id="cb22-2"><a href="#cb22-2" aria-hidden="true" tabindex="-1"></a>df_def <span class="op">=</span> df.copy()</span>
<span id="cb22-3"><a href="#cb22-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb22-4"><a href="#cb22-4" aria-hidden="true" tabindex="-1"></a><span class="co"># Droping non requiered features</span></span>
<span id="cb22-5"><a href="#cb22-5" aria-hidden="true" tabindex="-1"></a>df_def <span class="op">=</span> df_def.drop([<span class="st">&#39;nameOrig&#39;</span>, <span class="st">&#39;nameDest&#39;</span>, <span class="st">&#39;isFlaggedFraud&#39;</span>], axis <span class="op">=</span> <span class="dv">1</span>)</span>
<span id="cb22-6"><a href="#cb22-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb22-7"><a href="#cb22-7" aria-hidden="true" tabindex="-1"></a><span class="co"># Encoding categorical variables</span></span>
<span id="cb22-8"><a href="#cb22-8" aria-hidden="true" tabindex="-1"></a>df_def <span class="op">=</span> pd.get_dummies(df_def)</span></code></pre></div>
</div>
<div class="cell code" data-execution_count="9" data-colab="{&quot;height&quot;:270,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="09Q8aB8ML7bU" data-outputId="ec75f21e-506f-4d05-d5f5-94a40a50b490">
<div class="sourceCode" id="cb23"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb23-1"><a href="#cb23-1" aria-hidden="true" tabindex="-1"></a><span class="co"># log transformation of skewed fetures</span></span>
<span id="cb23-2"><a href="#cb23-2" aria-hidden="true" tabindex="-1"></a>sfeature <span class="op">=</span> [<span class="st">&#39;amount&#39;</span>,<span class="st">&#39;oldbalanceOrg&#39;</span>, <span class="st">&#39;newbalanceOrig&#39;</span>, <span class="st">&#39;oldbalanceDest&#39;</span>, <span class="st">&#39;newbalanceDest&#39;</span>]</span>
<span id="cb23-3"><a href="#cb23-3" aria-hidden="true" tabindex="-1"></a>df_def[sfeature] <span class="op">=</span> np.log(df_def[sfeature] <span class="op">+</span> <span class="dv">1</span>)</span>
<span id="cb23-4"><a href="#cb23-4" aria-hidden="true" tabindex="-1"></a>df_def.head()</span></code></pre></div>
<div class="output execute_result" data-execution_count="9">

  <div id="df-3396e7dc-4825-4776-a1d3-59b0b0905f0f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>type_CASH_IN</th>
      <th>type_CASH_OUT</th>
      <th>type_DEBIT</th>
      <th>type_PAYMENT</th>
      <th>type_TRANSFER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9.194276</td>
      <td>12.044359</td>
      <td>11.984786</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.531166</td>
      <td>9.964112</td>
      <td>9.872292</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5.204007</td>
      <td>5.204007</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5.204007</td>
      <td>5.204007</td>
      <td>0.000000</td>
      <td>9.960954</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>9.364703</td>
      <td>10.634773</td>
      <td>10.305174</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3396e7dc-4825-4776-a1d3-59b0b0905f0f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3396e7dc-4825-4776-a1d3-59b0b0905f0f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3396e7dc-4825-4776-a1d3-59b0b0905f0f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
</div>
</div>
<section id="model-pipeline" class="cell markdown" id="GgxuDO7LJD9a">
<h3>Model Pipeline</h3>
</section>
<div class="cell code" data-execution_count="10" data-colab="{&quot;height&quot;:1000,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="QGcUqW1pDuyr" data-outputId="42299fb6-54e1-4d7a-d949-73f2030b8482">
<div class="sourceCode" id="cb24"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb24-1"><a href="#cb24-1" aria-hidden="true" tabindex="-1"></a><span class="co"># features and target</span></span>
<span id="cb24-2"><a href="#cb24-2" aria-hidden="true" tabindex="-1"></a>features <span class="op">=</span> df_def.drop(<span class="st">&#39;isFraud&#39;</span>, axis <span class="op">=</span> <span class="va">True</span>)</span>
<span id="cb24-3"><a href="#cb24-3" aria-hidden="true" tabindex="-1"></a>target <span class="op">=</span> df_def[<span class="st">&#39;isFraud&#39;</span>]</span>
<span id="cb24-4"><a href="#cb24-4" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb24-5"><a href="#cb24-5" aria-hidden="true" tabindex="-1"></a><span class="co"># Splitting the train and test data</span></span>
<span id="cb24-6"><a href="#cb24-6" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.model_selection <span class="im">import</span> train_test_split</span>
<span id="cb24-7"><a href="#cb24-7" aria-hidden="true" tabindex="-1"></a>xtrain, xtest, ytrain, ytest <span class="op">=</span> train_test_split(features, target, test_size <span class="op">=</span> <span class="fl">0.3</span>, random_state <span class="op">=</span> <span class="dv">0</span>)</span>
<span id="cb24-8"><a href="#cb24-8" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb24-9"><a href="#cb24-9" aria-hidden="true" tabindex="-1"></a><span class="co"># Importing the classifiers</span></span>
<span id="cb24-10"><a href="#cb24-10" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.linear_model <span class="im">import</span> LogisticRegression</span>
<span id="cb24-11"><a href="#cb24-11" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.ensemble <span class="im">import</span> RandomForestClassifier</span>
<span id="cb24-12"><a href="#cb24-12" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.naive_bayes <span class="im">import</span> GaussianNB</span>
<span id="cb24-13"><a href="#cb24-13" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb24-14"><a href="#cb24-14" aria-hidden="true" tabindex="-1"></a><span class="co"># Performace metircs</span></span>
<span id="cb24-15"><a href="#cb24-15" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.metrics <span class="im">import</span> accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report</span>
<span id="cb24-16"><a href="#cb24-16" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb24-17"><a href="#cb24-17" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> performace(xtrain, ytrain, xtest, ytest, classifier):</span>
<span id="cb24-18"><a href="#cb24-18" aria-hidden="true" tabindex="-1"></a>  ypred <span class="op">=</span> classifier.predict(xtest)</span>
<span id="cb24-19"><a href="#cb24-19" aria-hidden="true" tabindex="-1"></a>  report <span class="op">=</span> classification_report(ytest,ypred)</span>
<span id="cb24-20"><a href="#cb24-20" aria-hidden="true" tabindex="-1"></a>  cm <span class="op">=</span> confusion_matrix(ytest,ypred)</span>
<span id="cb24-21"><a href="#cb24-21" aria-hidden="true" tabindex="-1"></a>  <span class="bu">print</span>(report)</span>
<span id="cb24-22"><a href="#cb24-22" aria-hidden="true" tabindex="-1"></a>  disp <span class="op">=</span> ConfusionMatrixDisplay(confusion_matrix <span class="op">=</span> cm)</span>
<span id="cb24-23"><a href="#cb24-23" aria-hidden="true" tabindex="-1"></a>  disp.plot()</span>
<span id="cb24-24"><a href="#cb24-24" aria-hidden="true" tabindex="-1"></a>  plt.show()</span>
<span id="cb24-25"><a href="#cb24-25" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb24-26"><a href="#cb24-26" aria-hidden="true" tabindex="-1"></a>  <span class="co"># Result</span></span>
<span id="cb24-27"><a href="#cb24-27" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> classifier <span class="kw">in</span> [LogisticRegression(random_state <span class="op">=</span> <span class="dv">0</span>), RandomForestClassifier(random_state <span class="op">=</span> <span class="dv">0</span>), GaussianNB()]:</span>
<span id="cb24-28"><a href="#cb24-28" aria-hidden="true" tabindex="-1"></a>  classifier.fit(xtrain, ytrain)</span>
<span id="cb24-29"><a href="#cb24-29" aria-hidden="true" tabindex="-1"></a>  <span class="bu">print</span>(classifier)</span>
<span id="cb24-30"><a href="#cb24-30" aria-hidden="true" tabindex="-1"></a>  performace(xtrain, ytrain, xtest, ytest, classifier)</span></code></pre></div>
<div class="output stream stdout">
<pre><code>LogisticRegression(random_state=0)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906367
           1       0.86      0.48      0.62      2419

    accuracy                           1.00   1908786
   macro avg       0.93      0.74      0.81   1908786
weighted avg       1.00      1.00      1.00   1908786

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/dbb3b0f280ae4c4c332d1e57cbdcb57da373431e.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>RandomForestClassifier(random_state=0)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906367
           1       0.98      0.79      0.88      2419

    accuracy                           1.00   1908786
   macro avg       0.99      0.90      0.94   1908786
weighted avg       1.00      1.00      1.00   1908786

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/e715f75a603fd2274ebbcd4889994628b40b8c79.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>GaussianNB()
              precision    recall  f1-score   support

           0       1.00      0.82      0.90   1906367
           1       0.01      0.97      0.01      2419

    accuracy                           0.82   1908786
   macro avg       0.50      0.89      0.46   1908786
weighted avg       1.00      0.82      0.90   1908786

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/add3ca905f1a8b942c39982afe67c7600cf79821.png" /></p>
</div>
</div>
<div class="cell markdown" id="Krpx9aI9HfnA">
<ul>
<li><p>It is quite clear that in the case of every classifier used above the classification for 0 label (non-fraud transaction) is very good. But in the opposite the classification of class label 1 (fraud transactions) is very bad, which is not acceptable.</p></li>
<li><p>For example in the case of Logistic regression classifier (f-1 scores = 0.62), 2419 samples are actually labeled 1. But out of 2419 samples, 1253 (more than half) are misclassified. But this situation is not happening for class zero.</p></li>
<li><p>Gaussian Naive Bayes classifier has classified class 1 with a very high accuracy but it also has classified many lables as 1 which were 0 in actual. Because of this the f-1 score has reduced to 0.01 which is much much lower than that of other two classifier.</p></li>
</ul>
</div>
<section id="classification-with-resampling" class="cell markdown" id="6TWCdjyNGu2L">
<h1>Classification with resampling</h1>
</section>
<section id="resampling" class="cell markdown" id="SkPovVM-JP1L">
<h2>Resampling</h2>
</section>
<div class="cell markdown" id="0zaKcwygGz3v">
<p>Resampling can be done in two ways:</p>
<ul>
<li><p>Over sampling - Creating duplicates of class 1 to make number of both the class samples equal. Therefore, total number of rows will be the twice of the samples supporting class 0 (i.e., more than original number of samples)</p></li>
<li><p>Under sampling - Dropping random samples supporting class 0 to make number of both the class samples equal. Therefore, total number of rows will be the twice of the samples supporting class 1 (i.e., Much less than the original number of samples)</p></li>
<li><p>As the dataframe has a lot of rows, Undersampling will be a better choice.</p></li>
</ul>
</div>
<section id="undersampling-resampling" class="cell markdown" id="iS8NPkYHH4uw">
<h2>Undersampling (Resampling)</h2>
</section>
<div class="cell code" id="_NGCLWWqH2zh">
<div class="sourceCode" id="cb28"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb28-1"><a href="#cb28-1" aria-hidden="true" tabindex="-1"></a><span class="co"># selecting the class 1 samples</span></span>
<span id="cb28-2"><a href="#cb28-2" aria-hidden="true" tabindex="-1"></a>df1 <span class="op">=</span> df[df[<span class="st">&#39;isFraud&#39;</span>] <span class="op">==</span> <span class="dv">1</span>]</span>
<span id="cb28-3"><a href="#cb28-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-4"><a href="#cb28-4" aria-hidden="true" tabindex="-1"></a><span class="co"># Selectiong the equal number of random class 0 samples</span></span>
<span id="cb28-5"><a href="#cb28-5" aria-hidden="true" tabindex="-1"></a>df0 <span class="op">=</span> df[df[<span class="st">&#39;isFraud&#39;</span>] <span class="op">==</span> <span class="dv">0</span>].sample(df1.shape[<span class="dv">0</span>])</span>
<span id="cb28-6"><a href="#cb28-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-7"><a href="#cb28-7" aria-hidden="true" tabindex="-1"></a><span class="co"># Merging to get the final data framd</span></span>
<span id="cb28-8"><a href="#cb28-8" aria-hidden="true" tabindex="-1"></a>df_und_sam <span class="op">=</span> pd.concat([df1, df0])</span>
<span id="cb28-9"><a href="#cb28-9" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-10"><a href="#cb28-10" aria-hidden="true" tabindex="-1"></a><span class="co"># Saving it for future refrence</span></span>
<span id="cb28-11"><a href="#cb28-11" aria-hidden="true" tabindex="-1"></a>df_und_sam.to_csv(<span class="st">&#39;Under_sampled_data.csv&#39;</span>)</span></code></pre></div>
</div>
<div class="cell code" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="vVGCxgD9OnWg" data-outputId="c90aeac0-3345-4459-ec3e-173cdbe2d133">
<div class="sourceCode" id="cb29"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb29-1"><a href="#cb29-1" aria-hidden="true" tabindex="-1"></a>df <span class="op">=</span> pd.read_csv(<span class="st">&#39;Under_sampled_data.csv&#39;</span>)</span>
<span id="cb29-2"><a href="#cb29-2" aria-hidden="true" tabindex="-1"></a>df.info()</span></code></pre></div>
<div class="output stream stdout">
<pre><code>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 16426 entries, 0 to 16425
Data columns (total 12 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Unnamed: 0      16426 non-null  int64  
 1   step            16426 non-null  int64  
 2   type            16426 non-null  object 
 3   amount          16426 non-null  float64
 4   nameOrig        16426 non-null  object 
 5   oldbalanceOrg   16426 non-null  float64
 6   newbalanceOrig  16426 non-null  float64
 7   nameDest        16426 non-null  object 
 8   oldbalanceDest  16426 non-null  float64
 9   newbalanceDest  16426 non-null  float64
 10  isFraud         16426 non-null  int64  
 11  isFlaggedFraud  16426 non-null  int64  
dtypes: float64(5), int64(4), object(3)
memory usage: 1.5+ MB
</code></pre>
</div>
</div>
<div class="cell markdown" id="PAr2YUT_7QT3">
<ul>
<li>Total number of samples in resampled dataframe 16426 which is twice of the number of samples supporting class 1.</li>
</ul>
</div>
<div class="cell code" data-colab="{&quot;height&quot;:345,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="nM_CRg99OHGo" data-outputId="281da0ab-9d04-484b-f771-b80312a06f39">
<div class="sourceCode" id="cb31"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb31-1"><a href="#cb31-1" aria-hidden="true" tabindex="-1"></a><span class="co"># General view of the data</span></span>
<span id="cb31-2"><a href="#cb31-2" aria-hidden="true" tabindex="-1"></a>plt.style.use(<span class="st">&#39;dark_background&#39;</span>)</span>
<span id="cb31-3"><a href="#cb31-3" aria-hidden="true" tabindex="-1"></a>fig <span class="op">=</span> plt.gcf()</span>
<span id="cb31-4"><a href="#cb31-4" aria-hidden="true" tabindex="-1"></a>fig.set_size_inches(<span class="dv">5</span>, <span class="dv">5</span>)</span>
<span id="cb31-5"><a href="#cb31-5" aria-hidden="true" tabindex="-1"></a>sns.scatterplot(x <span class="op">=</span> <span class="st">&#39;oldbalanceOrg&#39;</span>, y <span class="op">=</span> <span class="st">&#39;newbalanceDest&#39;</span>, data <span class="op">=</span> df[df[<span class="st">&#39;isFraud&#39;</span>] <span class="op">==</span> <span class="dv">1</span>], color <span class="op">=</span> <span class="st">&#39;red&#39;</span>, label <span class="op">=</span> <span class="st">&#39;Fraud&#39;</span>, marker <span class="op">=</span> <span class="st">&#39;*&#39;</span>)<span class="op">;</span></span>
<span id="cb31-6"><a href="#cb31-6" aria-hidden="true" tabindex="-1"></a>sns.scatterplot(x <span class="op">=</span> <span class="st">&#39;oldbalanceOrg&#39;</span>, y <span class="op">=</span> <span class="st">&#39;newbalanceDest&#39;</span>, data <span class="op">=</span> df[df[<span class="st">&#39;isFraud&#39;</span>] <span class="op">==</span> <span class="dv">0</span>], color <span class="op">=</span> <span class="st">&#39;green&#39;</span>, label <span class="op">=</span> <span class="st">&#39;non-Fraud&#39;</span>, marker <span class="op">=</span> <span class="st">&#39;+&#39;</span>)<span class="op">;</span></span>
<span id="cb31-7"><a href="#cb31-7" aria-hidden="true" tabindex="-1"></a>plt.ylim(<span class="op">-</span><span class="fl">0.01</span><span class="op">*</span>(<span class="dv">10</span><span class="op">**</span><span class="dv">8</span>), <span class="fl">0.5</span><span class="op">*</span>(<span class="dv">10</span><span class="op">**</span><span class="dv">8</span>))</span>
<span id="cb31-8"><a href="#cb31-8" aria-hidden="true" tabindex="-1"></a>plt.legend()</span>
<span id="cb31-9"><a href="#cb31-9" aria-hidden="true" tabindex="-1"></a>plt.show()</span></code></pre></div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/dce8e14723bbc4bdb3444960dd44c0e7087d9683.png" /></p>
</div>
</div>
<div class="cell markdown" id="zvYFQipi7fdD">
<ul>
<li>Now visual of fraud transactions is much clear compared to earlier.</li>
</ul>
</div>
<section id="visualization" class="cell markdown" id="KWEe55IO8OcP">
<h2>Visualization</h2>
</section>
<div class="cell code" data-colab="{&quot;height&quot;:291,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="nY65fGtYU1qD" data-outputId="5b7d01b7-76f9-4ebe-fc25-4ae5ebafb442">
<div class="sourceCode" id="cb32"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb32-1"><a href="#cb32-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Amount Vs transaction type</span></span>
<span id="cb32-2"><a href="#cb32-2" aria-hidden="true" tabindex="-1"></a>sns.swarmplot(x <span class="op">=</span> <span class="st">&#39;type&#39;</span>, y <span class="op">=</span> <span class="st">&#39;amount&#39;</span>, data <span class="op">=</span> df.sample(<span class="dv">1000</span>), hue<span class="op">=</span><span class="st">&#39;isFraud&#39;</span>)<span class="op">;</span></span></code></pre></div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/46c766774ea8cefbebda709af16193254d77cd74.png" /></p>
</div>
</div>
<div class="cell markdown" id="_999aCCD7w1m">
<ul>
<li>If we can clearly see that the most of the fraud transactions are either done by cashing it out or by transfering to another account.</li>
</ul>
</div>
<div class="cell code" data-colab="{&quot;height&quot;:1000,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="k2Uk69m9Wt9N" data-outputId="762d7e5a-7924-45ed-d28a-f163c6242f43">
<div class="sourceCode" id="cb33"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb33-1"><a href="#cb33-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Skewness in the numerical features</span></span>
<span id="cb33-2"><a href="#cb33-2" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> col <span class="kw">in</span> [<span class="st">&#39;amount&#39;</span>,<span class="st">&#39;oldbalanceOrg&#39;</span>, <span class="st">&#39;newbalanceOrig&#39;</span>, <span class="st">&#39;oldbalanceDest&#39;</span>, <span class="st">&#39;newbalanceDest&#39;</span>]:</span>
<span id="cb33-3"><a href="#cb33-3" aria-hidden="true" tabindex="-1"></a>  sns.boxenplot(x<span class="op">=</span>col, data <span class="op">=</span> df)<span class="op">;</span></span>
<span id="cb33-4"><a href="#cb33-4" aria-hidden="true" tabindex="-1"></a>  plt.show()</span>
<span id="cb33-5"><a href="#cb33-5" aria-hidden="true" tabindex="-1"></a>  <span class="bu">print</span>(col,<span class="st">&#39;</span><span class="ch">\n</span><span class="st">&#39;</span>)</span></code></pre></div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/0301e1f95af474e6afb6dd5b14728c0e27e75785.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>amount 

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/2ef2055eeaf56574055af6fd76d4c8f9df27d0cc.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>oldbalanceOrg 

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/4f728514c2677ad0f83bed322e53ea198c2183c2.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>newbalanceOrig 

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/744f84dc0386d2366c38ac59f198de5e349de6dd.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>oldbalanceDest 

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/d37bee9f17751337a118767adb0fdb459422f290.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>newbalanceDest 

</code></pre>
</div>
</div>
<section id="data-processing" class="cell markdown" id="w9C3nSySWyfj">
<h2>Data processing</h2>
</section>
<div class="cell code" id="8WXSRgwSW1_7">
<div class="sourceCode" id="cb39"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb39-1"><a href="#cb39-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Droping non requiered features</span></span>
<span id="cb39-2"><a href="#cb39-2" aria-hidden="true" tabindex="-1"></a>df <span class="op">=</span> df.drop([<span class="st">&#39;nameOrig&#39;</span>, <span class="st">&#39;nameDest&#39;</span>, <span class="st">&#39;isFlaggedFraud&#39;</span>], axis <span class="op">=</span> <span class="dv">1</span>)</span>
<span id="cb39-3"><a href="#cb39-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb39-4"><a href="#cb39-4" aria-hidden="true" tabindex="-1"></a><span class="co"># Encoding categorical variables</span></span>
<span id="cb39-5"><a href="#cb39-5" aria-hidden="true" tabindex="-1"></a>df <span class="op">=</span> pd.get_dummies(df)</span>
<span id="cb39-6"><a href="#cb39-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb39-7"><a href="#cb39-7" aria-hidden="true" tabindex="-1"></a><span class="co"># log transformation of skewed fetures</span></span>
<span id="cb39-8"><a href="#cb39-8" aria-hidden="true" tabindex="-1"></a>sfeature <span class="op">=</span> [<span class="st">&#39;amount&#39;</span>,<span class="st">&#39;oldbalanceOrg&#39;</span>, <span class="st">&#39;newbalanceOrig&#39;</span>, <span class="st">&#39;oldbalanceDest&#39;</span>, <span class="st">&#39;newbalanceDest&#39;</span>]</span>
<span id="cb39-9"><a href="#cb39-9" aria-hidden="true" tabindex="-1"></a>df[sfeature] <span class="op">=</span> np.log(df[sfeature] <span class="op">+</span> <span class="dv">1</span>)</span></code></pre></div>
</div>
<section id="model-pipeling" class="cell markdown" id="qeIGcMtBX-GD">
<h2>Model Pipeling</h2>
</section>
<div class="cell code" data-colab="{&quot;height&quot;:1000,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="5XlNk4JLYCA0" data-outputId="bb511f9d-efb3-4332-dac7-e368fb8f0e29">
<div class="sourceCode" id="cb40"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb40-1"><a href="#cb40-1" aria-hidden="true" tabindex="-1"></a><span class="co"># features and target</span></span>
<span id="cb40-2"><a href="#cb40-2" aria-hidden="true" tabindex="-1"></a>features <span class="op">=</span> df.drop(<span class="st">&#39;isFraud&#39;</span>, axis <span class="op">=</span> <span class="va">True</span>)</span>
<span id="cb40-3"><a href="#cb40-3" aria-hidden="true" tabindex="-1"></a>target <span class="op">=</span> df[<span class="st">&#39;isFraud&#39;</span>]</span>
<span id="cb40-4"><a href="#cb40-4" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb40-5"><a href="#cb40-5" aria-hidden="true" tabindex="-1"></a><span class="co"># Splitting the train and test data</span></span>
<span id="cb40-6"><a href="#cb40-6" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.model_selection <span class="im">import</span> train_test_split</span>
<span id="cb40-7"><a href="#cb40-7" aria-hidden="true" tabindex="-1"></a>xtrain, xtest, ytrain, ytest <span class="op">=</span> train_test_split(features, target, test_size <span class="op">=</span> <span class="fl">0.3</span>, random_state <span class="op">=</span> <span class="dv">0</span>)</span>
<span id="cb40-8"><a href="#cb40-8" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb40-9"><a href="#cb40-9" aria-hidden="true" tabindex="-1"></a><span class="co"># Classifiers</span></span>
<span id="cb40-10"><a href="#cb40-10" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.linear_model <span class="im">import</span> LogisticRegression</span>
<span id="cb40-11"><a href="#cb40-11" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.ensemble <span class="im">import</span> RandomForestClassifier</span>
<span id="cb40-12"><a href="#cb40-12" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.naive_bayes <span class="im">import</span> GaussianNB</span>
<span id="cb40-13"><a href="#cb40-13" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb40-14"><a href="#cb40-14" aria-hidden="true" tabindex="-1"></a><span class="co"># Performace metircs</span></span>
<span id="cb40-15"><a href="#cb40-15" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.metrics <span class="im">import</span> accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report</span>
<span id="cb40-16"><a href="#cb40-16" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb40-17"><a href="#cb40-17" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> performace(xtrain, ytrain, xtest, ytest, classifier):</span>
<span id="cb40-18"><a href="#cb40-18" aria-hidden="true" tabindex="-1"></a>  ypred <span class="op">=</span> classifier.predict(xtest)</span>
<span id="cb40-19"><a href="#cb40-19" aria-hidden="true" tabindex="-1"></a>  report <span class="op">=</span> classification_report(ytest,ypred)</span>
<span id="cb40-20"><a href="#cb40-20" aria-hidden="true" tabindex="-1"></a>  cm <span class="op">=</span> confusion_matrix(ytest,ypred)</span>
<span id="cb40-21"><a href="#cb40-21" aria-hidden="true" tabindex="-1"></a>  <span class="bu">print</span>(report)</span>
<span id="cb40-22"><a href="#cb40-22" aria-hidden="true" tabindex="-1"></a>  disp <span class="op">=</span> ConfusionMatrixDisplay(confusion_matrix <span class="op">=</span> cm)</span>
<span id="cb40-23"><a href="#cb40-23" aria-hidden="true" tabindex="-1"></a>  disp.plot()</span>
<span id="cb40-24"><a href="#cb40-24" aria-hidden="true" tabindex="-1"></a>  plt.show()</span>
<span id="cb40-25"><a href="#cb40-25" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb40-26"><a href="#cb40-26" aria-hidden="true" tabindex="-1"></a><span class="co"># Result</span></span>
<span id="cb40-27"><a href="#cb40-27" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> classifier <span class="kw">in</span> [LogisticRegression(random_state <span class="op">=</span> <span class="dv">0</span>), RandomForestClassifier(random_state <span class="op">=</span> <span class="dv">0</span>), GaussianNB()]:</span>
<span id="cb40-28"><a href="#cb40-28" aria-hidden="true" tabindex="-1"></a>  classifier.fit(xtrain, ytrain)</span>
<span id="cb40-29"><a href="#cb40-29" aria-hidden="true" tabindex="-1"></a>  <span class="bu">print</span>(classifier)</span>
<span id="cb40-30"><a href="#cb40-30" aria-hidden="true" tabindex="-1"></a>  performace(xtrain, ytrain, xtest, ytest, classifier)</span></code></pre></div>
<div class="output stream stdout">
<pre><code>LogisticRegression(random_state=0)
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      2490
           1       0.49      1.00      0.66      2438

    accuracy                           0.49      4928
   macro avg       0.25      0.50      0.33      4928
weighted avg       0.24      0.49      0.33      4928

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/fd6f3906b43725452820a8888061a5a217cd5d19.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>
RandomForestClassifier(random_state=0)
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      2490
           1       0.99      0.99      0.99      2438

    accuracy                           0.99      4928
   macro avg       0.99      0.99      0.99      4928
weighted avg       0.99      0.99      0.99      4928

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/67323a66c501ee051689e591a477c68a54e341b5.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>
GaussianNB()
              precision    recall  f1-score   support

           0       0.63      0.74      0.68      2490
           1       0.68      0.56      0.62      2438

    accuracy                           0.65      4928
   macro avg       0.66      0.65      0.65      4928
weighted avg       0.66      0.65      0.65      4928

</code></pre>
</div>
<div class="output display_data">
<p><img src="vertopal_cee06adbe43041f18c270e855e7b44fe/cb98e16d2d045a976a0dd5bfcf5bbc016f24fda9.png" /></p>
</div>
<div class="output stream stdout">
<pre><code>
</code></pre>
</div>
</div>
<section id="conclusion" class="cell markdown" id="5WeXE5FgYqTR">
<h1>Conclusion</h1>
</section>
<div class="cell markdown" id="mpbsrLg6Yxip">
<ul>
<li><p>Out of three classfiers, Random Forest classifier is working best before and after reshampling. It is also clear that it is working much better after resampling.</p></li>
<li><p>Logistic regression classifier was baised towards the 0 leble before resampling and it became biased towards label 1 after resampling (under sampling leads to loose a lot of valuable information)</p></li>
<li><p>Gaussian Naive Bayes classifier's is performing better after resampling.</p></li>
<li><p>f1-score of every classifier is increased for every classifier.</p></li>
<li><p>As this is the case of fraud detection, detecting of a fraud transactions as non-fraud is worse than detecting of a non-fraud transacation as fraud.</p></li>
</ul>
</div>
