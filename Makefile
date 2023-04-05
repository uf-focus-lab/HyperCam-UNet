MAKE ?= make
MAKE := $(MAKE) --no-print-directory
E    := \033[
RUNS ?= $(patsubst var/run/%,%,$(wildcard var/run/*))
DIRS := $(patsubst %,var/run/%,$(RUNS))

log.stat:
	$(eval RUNS:=$(shell cat var/run.log | awk '{ print $$1; }' | xargs))
	@	RUNS="$(RUNS)" $(MAKE) stat

stat:
	@	for i in $(DIRS)			\
	;do	printf "$$(basename $$i) "	\
	;	test -f $$i/000_SUCCESS		\
	&&	test -f $$i/train.log.txt	\
	&&	(cat $$i/train.log.txt | wc -l)	\
	||	echo "$E0;31munsuccessful$E0m"	\
	;	done
clean:
	@mkdir -p var/failed
	@	for i in $(DIRS); do		\
		test -f $$i/000_SUCCESS		\
	||	mv "$$i" "$$i/../failed";	\
		done

.PHONY: log.stat stat clean