default:
	@ echo "What do you want to make?"
	@ echo ""
	@ echo "  manuscript"

.PHONY: manuscript
manuscript:
	@ cd manuscript; make && cd -
