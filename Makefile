push: 
	@git status 
	git add .
	git commit -m "Dp: $(yx)"
	git push
