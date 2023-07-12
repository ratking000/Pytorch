# Pytorch
本篇是學習紀錄，為了防止我忘記之前當下想到的問題，也希望我pytorch可以持續進步  

### 0712 record  
train01是我自己湊出來的，02則是範例  
目前01的正確率和loss都表現較佳  

但還是想看到為什麼01與02的acc和loss不一樣，畢竟都長得差不多  
目前看到的差異是  
1. train02的97行model.eval()，不知是幹啥的  
2. train02中train(train_dataloader, model, loss_fn, optimizer) 在traain01沒有loss_fn，但感覺沒差  

但還有幾點問題是  
1. 好像每次跑的acc和loss都不太一樣，是我要給他一個種子碼讓他固定train的dataset  嗎?還是只是我epochs給太少  
2. 我如果沒在train和test中打X, y = X.to(device), y.to(device)好像就會有問題，錯誤是說我cpu gpu都有吃到東西，正常只能在其中一個中執行，這問題沒辦法一開始就設定好我要用什麼嗎?不然我每次都要重打X, y = X.to(device), y.to(device)  

目前大概流程已經懂了，之後的學習方向可以朝
1. 為啥model長那樣，是有什麼可以做參考的點嗎  
2. train test中有很多東西不知道是怎麼叫出來的(例如test_loss += loss_fn(pred, y).item()等等)，接下來慢慢熟悉  
3. 練習更多模型  
