#!/usr/bin/env node
import { Command } from 'commander'
import { setOutputFormat } from './lib/output'

const program = new Command()
  .name('cli')
  .description('Orchestral CLI')
  .version('0.1.0')
  .option('--json', 'Output as JSON (machine-readable)')
  .hook('preAction', (thisCommand: Command) => {
    const opts = thisCommand.opts()
    if (opts.json) {
      setOutputFormat('json')
    }
  })

// Commands will be added as features are built

// Show help if no command provided
if (process.argv.length === 2) {
  program.help()
}

program.parse()
