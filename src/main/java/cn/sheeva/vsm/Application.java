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
            
            List<String> tokens=tokenizer.getTokens(FileUtils.readFileToString(f));
            for (Entry<String, List<Doc>>  entry: docs.entrySet()) {
                String category=entry.getKey();
                List<Doc> categoryDocs=entry.getValue();
                double total=0;
                for (Doc doc : categoryDocs) {
                    double score=doc.calculateSimilarity(tokens);
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
    
    
    
    private static class Doc{
        private HashMap<String, Long> tfs=null;
        private HashMap<String, Double> tfidfs=new HashMap<>();
        
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
        
        public double calculateSimilarity(List<String> tokens){
            double total=0;
            for (String token : tokens) {
                Double tfidf=tfidfs.get(token);
                if (tfidf!=null) {
                    total+=tfidf;
                }
            }
            return total;
        }
    }
    
    
}
