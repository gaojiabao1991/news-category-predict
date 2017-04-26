package cn.sheeva.vsm;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.codec.binary.StringUtils;
import org.apache.commons.io.FileUtils;

import cn.sheeva.token.SimpleTokenizer;
import cn.sheeva.util.ResourceUtil;

public class Application {
    private static SimpleTokenizer tokenizer=new SimpleTokenizer();
    private static String dataDir=ResourceUtil.getResourcePath("corpus");
    private static String testDir=ResourceUtil.getResourcePath("testcase");
    
    
    public static void main(String[] args) throws IOException {
        Set<String> vocabularySet = getVocabularySet();
        HashMap<String, Long> dfs=new HashMap<>();
        
        HashMap<String, Long> initWordBag=new HashMap<>();
        for (String word : vocabularySet) {
            initWordBag.put(word, 0l);
        }
        
        for (String word : vocabularySet) {
            dfs.put(word, 0l);
        }
        
        //count tf and df
        HashMap<String, List<Doc>> docs=new HashMap<>();
        long docNum=0l;
        File root=new File(dataDir);
        for (File category : root.listFiles()) {
            List<Doc> categoryDocs=new LinkedList<>();
            for (File text : category.listFiles()) {
                System.out.println(text.getName());
                Doc doc=new Doc(initWordBag, FileUtils.readFileToString(text), dfs);
                categoryDocs.add(doc);
                
                docNum++;
            }
            docs.put(category.getName(), categoryDocs);
        }
        
        //calculate tfidf for each doc
        for (Entry<String, List<Doc>> entry : docs.entrySet()) {
            List<Doc> categoryDocs=entry.getValue();
            for (Doc doc : categoryDocs) {
                doc.calculateTfidf(dfs, docNum);
            }
        }
        
        //open test doc and calculate similarity
        
        File testRoot=new File(testDir);
        for (File f : testRoot.listFiles()) {
            double maxScore=0;
            String predictCategory=null;
            System.out.println("文章名称: "+f.getName());
            TestDoc testDoc=new TestDoc(initWordBag, FileUtils.readFileToString(f), dfs, docNum);
            
            for (Entry<String, List<Doc>>  entry: docs.entrySet()) {
                String category=entry.getKey();
                List<Doc> categoryDocs=entry.getValue();
                double total=0;
                for (Doc doc : categoryDocs) {
                    double score=testDoc.calculateSimilarity(doc);
                    total+=score;
                }
                
                System.out.println("类目："+category+",\t得分："+total);
                if (total>maxScore) {
                    maxScore=total;
                    predictCategory=category;
                }
            }
            System.out.println("预测类目："+predictCategory+"\n");
        }
        
    }
    

    private static Set<String> getVocabularySet() throws IOException {
        Set<String> vocabularySet=new HashSet<>();
        File root=new File(dataDir);
        for (File category : root.listFiles()) {
            for (File text : category.listFiles()) {
                System.out.println(text.getName());
                String textStr=FileUtils.readFileToString(text, "utf-8");
                vocabularySet.addAll(tokenizer.getTokens(textStr));
                
            }
        }
        return vocabularySet;
    }
        
    private static class TestDoc{
        private HashMap<String, Long> tfs=null;
        private HashMap<String, Long> dfs=null;
        private HashMap<String, Double> tfidfs=new HashMap<>();
        private long docNum;
        
        public TestDoc(HashMap<String, Long> initWordBag,String text,HashMap<String, Long> dfs, long docNum) {
            tfs=new HashMap<>(initWordBag);
            this.dfs=dfs;
            this.docNum=docNum;
            
            List<String> tokens=tokenizer.getTokens(text);
            for (String token : tokens) {
                Long tf=tfs.get(token);
                if (tf!=null) {
                    tf++;
                    tfs.put(token, tf);
                }
            }
            
            this.calculateTfidf();
        }
        
        private void calculateTfidf(){
            for (Entry<String, Long> entry : tfs.entrySet()) {
                String word=entry.getKey();
                long tf=tfs.get(word);
                long df=dfs.get(word);
                double tfidf=tf*Math.log(docNum/df);
                tfidfs.put(word, tfidf);
            }
        }
        
        public double calculateSimilarity(Doc doc){
            if (this.tfidfs.size()!=doc.tfidfs.size()) {
                throw new RuntimeException("Doc向量维度不一致！");
            }
            double dotProduct=0d;
            for (Entry<String, Double> entry : this.tfidfs.entrySet()) {
                String word=entry.getKey();
                double tfidf1=this.tfidfs.get(word);
                double tfidf2=doc.tfidfs.get(word);
                dotProduct+=tfidf1*tfidf2;
            }
            
            double OlenSquare1=0d;
            for (Entry<String, Double> entry : this.tfidfs.entrySet()) {
                double tfidf=entry.getValue();
                OlenSquare1+=tfidf*tfidf;
            }
            
            double OlenSquare2=0d;
            for (Entry<String, Double> entry : doc.tfidfs.entrySet()) {
                double tfidf=entry.getValue();
                OlenSquare2+=tfidf*tfidf;
            }
            
            double score=dotProduct/(Math.sqrt(OlenSquare1)*Math.sqrt(OlenSquare2));
            return score;
        }
    }
    
    
    
    private static class Doc{
        private HashMap<String, Long> tfs=null;
        public HashMap<String, Double> tfidfs=new HashMap<>();
        
        public Doc(HashMap<String, Long> initWordBag,String text,HashMap<String,Long> dfs) {
            this.tfs=new HashMap<>(initWordBag);
            List<String> words=tokenizer.getTokens(text);
            for (String word : words) {
                Long c=tfs.get(word);
                c++;
                if (c==1) {
                    long idf=dfs.get(word)+1l;
                    dfs.put(word, idf);
                }
                tfs.put(word, c);
            }
        }
        
        public void calculateTfidf(HashMap<String, Long> dfs, long docNum){
            for (Entry<String, Long> entry : tfs.entrySet()) {
                String word=entry.getKey();
                long tf=tfs.get(word);
                long df=dfs.get(word);
                double tfidf=tf*Math.log(docNum/df);
                tfidfs.put(word, tfidf);
            }
        }
        

    }
    
    
}
