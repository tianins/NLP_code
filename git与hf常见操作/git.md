## git

1. 登录

   ```
   git config --global user.email "tianins@github.com"
   
   git config --global user.name "tianins"
   
   ssh-keygen -t rsa -b 4096 -C "tianins@github.com"
   
   cat ~/.ssh/id_rsa.pub
   
   ssh -T git@github.com
   ```

   

2. git操作

   ```
   git init 初始化
   
   git status 查看缓存区内容
   
   git rm --cached <文件>
   
   .gitignore 只能忽略那些原来没有被跟踪的文件,如果某些文件已经被纳入了版本管理中,则修改 .gitignore 是  无效的。如果想忽略已被跟踪的文件,需要先用 git rm --cached <文件> 移出版本控制,然后再添加到 .gitignore 中。  
   
   将本地修改添加到暂存区
   git add .
   提交更新,撰写提交信息
   git commit -m "更新说明"
   
   git remote add origin git@github.com:tianins/llama-ft.git
   这行添加一个名为 origin 的远程仓库
   git branch -M main  
   将当前分支重命名为 main
   git push -u origin main git@github.com:tianins/llama-ft.git
   推送到远程仓库
   git push -f origin main
   ```

   

3. 安装lfs

   ```
   1.添加官方源
   curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
   2.通过apt安装
   apt install git-lfs
   ```

4. git下载

   ```
   下载直接使用git lfs clone xxxx
   
   git lfs install
   让一个普通 Git 仓库启用 Git LFS 来更好地管理大文件和二进制文件
   
   git lfs pull 
   的作用是从远程 LFS 服务器拉取版本控制中的大文件到本地,保证本地仓库包含完整的文件版本。
   
   git lfs fetch
   fetch 会将文件下载到本地缓存,不会自动更新工作区,pull 会直接下载到工作区。
   fetch 需要额外调用 git lfs checkout 更新工作区,pull 一步完成全部操作。
   git lfs fetch 只是下载更新本地缓存,不改动工作区文件。
   拉取代码后通常需要配合 fetch + checkout 来更新 LFS 文件
   
   git config --global http.sslVerify false
   ```

   

## hf

1. 登录

   ```
   huggingface-cli login
   输入网站的token：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
   再输入用户名：hqp
   密码：Weather4209.
   ```

   