		elif opt_name == 'random-inits':
			try:
				num_random_inits_ = std::stoul(opt_val)
				desired_num_random_inits_ = num_random_inits_

			except:
				raise Error('Invalid argument "' + opt_val + '" for option random-inits. Usage: options = "[--random-inits <convertible to int greater 0>]"')

			if num_random_inits_ <= 0:
				raise Error('Invalid argument "' + opt_val + '" for option random-inits. Usage: options = "[--random-inits <convertible to int greater 0>]"')

		}
		elif opt_name == 'randomness':
			if opt_val == 'PSEUDO':
				use_real_randomness_ = False

			elif opt_val == 'REAL':
				use_real_randomness_ = True

			else:
				raise Error('Invalid argument "' + opt_val  + '" for option randomness. Usage: options = "[--randomness REAL|PSEUDO] [...]"')

		}
		elif opt_name == 'stdout':
			if opt_val == '0':
				print_to_stdout_ = 0

			elif opt_val == '1':
				print_to_stdout_ = 1

			elif opt_val == '2':
				print_to_stdout_ = 2

			else:
				raise Error('Invalid argument "' + opt_val  + '" for option stdout. Usage: options = "[--stdout 0|1|2] [...]"')

		}
		elif opt_name == 'refine':
			if opt_val == 'TRUE':
				refine_ = True

			elif opt_val == 'FALSE':
				refine_ = False

			else:
				raise Error('Invalid argument "' + opt_val  + '" for option refine. Usage: options = "[--refine TRUE|FALSE] [...]"')

		}
		elif opt_name == 'time-limit':
			try:
				time_limit_in_sec_ = std::stod(opt_val)

			except:
				raise Error('Invalid argument "' + opt_val + '" for option time-limit.  Usage: options = "[--time-limit <convertible to double>] [...]')

		}
		elif opt_name == 'max-itrs':
			try:
				max_itrs_ = std::stoi(opt_val)

			except:
				raise Error('Invalid argument "' + opt_val + '" for option max-itrs. Usage: options = "[--max-itrs <convertible to int>] [...]')

		}
		elif opt_name == 'max-itrs-without-update':
			try:
				max_itrs_without_update_ = std::stoi(opt_val)

			except:
				raise Error('Invalid argument "' + opt_val + '" for option max-itrs-without-update. Usage: options = "[--max-itrs-without-update <convertible to int>] [...]')

		}
		elif opt_name == 'seed':
			try:
				seed_ = std::stoul(opt_val)

			except:
				raise Error('Invalid argument "' + opt_val + '" for option seed. Usage: options = "[--seed <convertible to int greater equal 0>] [...]')

		}
		elif opt_name == 'epsilon':
			try:
				epsilon_ = std::stod(opt_val)

			except:
				raise Error('Invalid argument "' + opt_val + '" for option epsilon. Usage: options = "[--epsilon <convertible to double greater 0>] [...]')

			if epsilon_ <= 0:
				raise Error('Invalid argument "' + opt_val + '" for option epsilon. Usage: options = "[--epsilon <convertible to double greater 0>] [...]')

		}
		elif opt_name == 'inits-increase-order':
			try:
				num_inits_increase_order_ = std::stoul(opt_val)

			except:
				raise Error('Invalid argument "' + opt_val + '" for option inits-increase-order. Usage: options = "[--inits-increase-order <convertible to int greater 0>]"')

			if num_inits_increase_order_ <= 0:
				raise Error('Invalid argument "' + opt_val + '" for option inits-increase-order. Usage: options = "[--inits-increase-order <convertible to int greater 0>]"')

		}
		elif opt_name == 'init-type-increase-order':
			init_type_increase_order_ = opt_val
			if opt_val != 'CLUSTERS' and opt_val != 'K-MEANS++':
				raise Exception(std::string('Invalid argument ') + opt_val + ' for option init-type-increase-order. Usage: options = "[--init-type-increase-order CLUSTERS|K-MEANS++] [...]"')

		}
		elif opt_name == 'max-itrs-increase-order':
			try:
				max_itrs_increase_order_ = std::stoi(opt_val)

			except:
				raise Error('Invalid argument "' + opt_val + '" for option max-itrs-increase-order. Usage: options = "[--max-itrs-increase-order <convertible to int>] [...]')

		}
		else:
			std::string valid_options('[--init-type <arg>] [--random-inits <arg>] [--randomness <arg>] [--seed <arg>] [--stdout <arg>] ')
			valid_options += '[--time-limit <arg>] [--max-itrs <arg>] [--epsilon <arg>] '
			valid_options += '[--inits-increase-order <arg>] [--init-type-increase-order <arg>] [--max-itrs-increase-order <arg>]'
			raise Error(std::string('Invalid option "') + opt_name + '". Usage: options = "' + valid_options + '"')

