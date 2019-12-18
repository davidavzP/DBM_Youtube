push: 
	@git status 
	git add .
	git commit -m "Dp: $(Dp)"
	git push